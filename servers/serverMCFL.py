import copy
from collections import defaultdict

import torch
import torch.nn.functional as F

from models.mcfl_models import MCFLClientEncoder
from utils.mcfl_clustering import agglomerative_cluster, kmeans_cluster
from utils.mcfl_utils import count_parameters


class MCFLServer:
    def __init__(
        self,
        global_model,
        num_clusters,
        encoder_embed_dim=32,
        outer_lr=1e-2,
        model_mix=0.5,
        device="cpu",
        total_rounds=None,
        recluster_every=1,
        recluster_warmup_rounds=0,
        stop_recluster_after=-1,
        skip_final_recluster=True,
        cluster_method="kmeans",
        cluster_feature="updates",
    ):
        self.device = device
        self.num_clusters = num_clusters
        self.outer_lr = outer_lr
        self.model_mix = model_mix
        self.total_rounds = total_rounds
        self.recluster_every = recluster_every
        self.recluster_warmup_rounds = recluster_warmup_rounds
        self.stop_recluster_after = stop_recluster_after
        self.skip_final_recluster = skip_final_recluster
        self.cluster_method = cluster_method
        self.cluster_feature = cluster_feature

        self.cluster_models = [
            copy.deepcopy(global_model).to(device)
            for _ in range(num_clusters)
        ]

        update_dim = count_parameters(global_model)
        self.encoder = MCFLClientEncoder(update_dim, encoder_embed_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

    def _sanitize_update_matrix(self, update_mat):
        finite_mask = torch.isfinite(update_mat)
        if finite_mask.all():
            return update_mat, 0

        cleaned = torch.nan_to_num(update_mat.detach(), nan=0.0, posinf=1e6, neginf=-1e6)
        cleaned = torch.clamp(cleaned, min=-1e6, max=1e6)
        bad_rows = (~finite_mask).any(dim=1).sum().item()
        return cleaned, int(bad_rows)

    def assign_initial_clusters(self, clients):
        for i, client in enumerate(clients):
            client.cluster_id = i % self.num_clusters

    def _build_cluster_points(self, update_mat):
        update_mat, bad_rows = self._sanitize_update_matrix(update_mat)
        if bad_rows > 0:
            print(f"[MCFL] Recluster sanitized {bad_rows} client update vectors with non-finite values.")

        normalized_updates = F.normalize(update_mat, dim=-1)

        if self.cluster_feature == "encoder":
            with torch.no_grad():
                embeddings = self.encoder(update_mat)
                if not torch.isfinite(embeddings).all():
                    embeddings = F.normalize(
                        torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6),
                        dim=-1,
                    )
                    print("[MCFL] Recluster sanitized non-finite encoder embeddings.")
            return embeddings.detach().cpu().numpy()

        return normalized_updates.detach().cpu().numpy()

    def _blend_cluster_models(self, cluster_to_params):
        if self.model_mix <= 0:
            return

        for cluster_id in range(self.num_clusters):
            if len(cluster_to_params[cluster_id]) == 0:
                continue

            model = self.cluster_models[cluster_id]
            total_weight = sum(weight for _, weight in cluster_to_params[cluster_id])
            if total_weight <= 0:
                total_weight = float(len(cluster_to_params[cluster_id]))

            averaged_params = {}
            for name in model.state_dict().keys():
                stacked = torch.stack(
                    [params[name].detach().to(self.device) * weight for params, weight in cluster_to_params[cluster_id]],
                    dim=0,
                )
                averaged_params[name] = stacked.sum(dim=0) / total_weight

            with torch.no_grad():
                state_dict = model.state_dict()
                for name, value in state_dict.items():
                    if name not in averaged_params:
                        continue
                    mixed = (1.0 - self.model_mix) * value + self.model_mix * averaged_params[name].to(value.device)
                    value.copy_(mixed)

    def _should_recluster(self, round_idx):
        current_round = round_idx + 1

        if self.recluster_every <= 0:
            return False
        if current_round < self.recluster_warmup_rounds:
            return False
        if current_round % self.recluster_every != 0:
            return False
        if self.stop_recluster_after > 0 and current_round > self.stop_recluster_after:
            return False
        if self.skip_final_recluster and self.total_rounds is not None and current_round >= self.total_rounds:
            return False
        return True

    def aggregate_meta_grads(self, cluster_to_grads):
        for cluster_id in range(self.num_clusters):
            if len(cluster_to_grads[cluster_id]) == 0:
                continue

            model = self.cluster_models[cluster_id]
            num_params = len(list(model.parameters()))
            total_weight = sum(weight for _, weight in cluster_to_grads[cluster_id])
            if total_weight <= 0:
                total_weight = float(len(cluster_to_grads[cluster_id]))

            avg_grads = []
            for p_idx in range(num_params):
                stacked = torch.stack(
                    [grads[p_idx].detach() * weight for grads, weight in cluster_to_grads[cluster_id]],
                    dim=0,
                )
                avg_grads.append(stacked.sum(dim=0) / total_weight)

            with torch.no_grad():
                for p, g in zip(model.parameters(), avg_grads):
                    p -= self.outer_lr * g

    def recluster_clients(self, clients, client_cluster_vecs):
        with torch.no_grad():
            update_mat = torch.stack(client_cluster_vecs, dim=0).to(self.device)
            cluster_points = self._build_cluster_points(update_mat)

        if self.cluster_method == "agglomerative":
            assignments = agglomerative_cluster(
                cluster_points,
                num_clusters=self.num_clusters,
            )
        else:
            assignments = kmeans_cluster(
                cluster_points,
                num_clusters=self.num_clusters,
                seed=42,
            )

        for client, cluster_id in zip(clients, assignments):
            client.cluster_id = int(cluster_id)

    def train_round(self, clients, round_idx, inner_lr=0.1, first_order=True, local_epochs=1):
        cluster_to_grads = defaultdict(list)
        cluster_to_params = defaultdict(list)
        client_cluster_vecs = []
        stats_list = []

        for client in clients:
            model = self.cluster_models[client.cluster_id]
            meta_grads, update_vec, head_update_vec, adapted_params, stats = client.local_adapt_and_meta_grad(
                model,
                inner_lr=inner_lr,
                first_order=first_order,
                local_epochs=local_epochs,
            )

            client_weight = max(int(stats["query_samples"]), 1)
            cluster_to_grads[client.cluster_id].append((meta_grads, client_weight))
            cluster_to_params[client.cluster_id].append((adapted_params, client_weight))
            client_cluster_vecs.append(head_update_vec if self.cluster_feature == "head_updates" else update_vec)
            stats_list.append(stats)

        self.aggregate_meta_grads(cluster_to_grads)
        self._blend_cluster_models(cluster_to_params)

        if self._should_recluster(round_idx):
            self.recluster_clients(clients, client_cluster_vecs)

        return stats_list
