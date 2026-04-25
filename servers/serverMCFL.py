import copy
from collections import defaultdict

import torch

from models.mcfl_models import MCFLClientEncoder
from utils.mcfl_clustering import kmeans_cluster
from utils.mcfl_utils import count_parameters


class MCFLServer:
    def __init__(
        self,
        global_model,
        num_clusters,
        encoder_embed_dim=32,
        outer_lr=1e-2,
        device="cpu",
        recluster_every=1,
    ):
        self.device = device
        self.num_clusters = num_clusters
        self.outer_lr = outer_lr
        self.recluster_every = recluster_every

        self.cluster_models = [
            copy.deepcopy(global_model).to(device)
            for _ in range(num_clusters)
        ]

        update_dim = count_parameters(global_model)
        self.encoder = MCFLClientEncoder(update_dim, encoder_embed_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

    def assign_initial_clusters(self, clients):
        for i, client in enumerate(clients):
            client.cluster_id = i % self.num_clusters

    def aggregate_meta_grads(self, cluster_to_grads):
        for cluster_id in range(self.num_clusters):
            if len(cluster_to_grads[cluster_id]) == 0:
                continue

            model = self.cluster_models[cluster_id]
            num_params = len(list(model.parameters()))

            avg_grads = []
            for p_idx in range(num_params):
                stacked = torch.stack(
                    [grads[p_idx].detach() for grads in cluster_to_grads[cluster_id]],
                    dim=0,
                )
                avg_grads.append(stacked.mean(dim=0))

            with torch.no_grad():
                for p, g in zip(model.parameters(), avg_grads):
                    p -= self.outer_lr * g

    def recluster_clients(self, clients, client_update_vecs):
        with torch.no_grad():
            update_mat = torch.stack(client_update_vecs, dim=0).to(self.device)
            embeddings = self.encoder(update_mat).detach().cpu().numpy()

        assignments = kmeans_cluster(
            embeddings,
            num_clusters=self.num_clusters,
            seed=42,
        )

        for client, cluster_id in zip(clients, assignments):
            client.cluster_id = int(cluster_id)

    def train_round(self, clients, round_idx, inner_lr=0.1, first_order=True, local_epochs=1):
        cluster_to_grads = defaultdict(list)
        client_update_vecs = []
        stats_list = []

        for client in clients:
            model = self.cluster_models[client.cluster_id]
            meta_grads, update_vec, stats = client.local_adapt_and_meta_grad(
                model,
                inner_lr=inner_lr,
                first_order=first_order,
                local_epochs=local_epochs,
            )

            cluster_to_grads[client.cluster_id].append(meta_grads)
            client_update_vecs.append(update_vec)
            stats_list.append(stats)

        self.aggregate_meta_grads(cluster_to_grads)

        if self.recluster_every > 0 and (round_idx + 1) % self.recluster_every == 0:
            self.recluster_clients(clients, client_update_vecs)

        return stats_list

