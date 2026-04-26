import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.cfl_federation import copy_state, reduce_add_average, flatten, pairwise_angles, pairwise_cluster_split


class CFLServer:
    def __init__(self, global_model, test_data=None, device="cpu"):
        self.device = device
        self.global_model = global_model.to(self.device)
        self.data = test_data
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False) if self.data is not None else None
        self.W = {key: value for key, value in self.global_model.named_parameters()}
        self.model_cache = []

    def synchronize_clients(self, clients):
        for client in clients:
            client.synchronize_with_server(self)

    def select_clients(self, clients, frac=1.0):
        if not clients:
            return []
        n_selected = max(1, int(len(clients) * frac))
        n_selected = min(len(clients), n_selected)
        return random.sample(list(clients), n_selected)

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])

    def cluster_clients(self, S):
        return pairwise_cluster_split(S)

    def aggregate_clusterwise(self, client_clusters, active_ids=None):
        for cluster in client_clusters:
            if not cluster:
                continue
            if active_ids is None:
                sources = [client.dW for client in cluster]
            else:
                sources = [client.dW for client in cluster if client.id in active_ids]
            if not sources:
                continue
            reduce_add_average(targets=[client.W for client in cluster], sources=sources)

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster]) if cluster else 0.0

    def compute_mean_update_norm(self, cluster):
        if not cluster:
            return 0.0
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        idcs = np.asarray(idcs, dtype=int)
        self.model_cache += [
            (
                idcs,
                {name: value.data.clone() if hasattr(value, "data") else value.clone() for name, value in params.items()},
                [accuracies[i] for i in idcs],
            )
        ]
