import copy
from collections import defaultdict

import torch
import torch.nn.functional as F

from models.mcfl_models import MCFLClientEncoder
from utils.mcfl_clustering import agglomerative_cluster, kmeans_cluster
from utils.mcfl_utils import count_parameters, sanitize_model_


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
        max_reclusters=-1,
        skip_final_recluster=True,
        cluster_change_threshold=0.1,
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
        self.max_reclusters = max_reclusters
        self.skip_final_recluster = skip_final_recluster
        self.cluster_change_threshold = cluster_change_threshold
        self.cluster_method = cluster_method
        self.cluster_feature = cluster_feature
        self.recluster_count = 0

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
            client_param_names = set.intersection(
                *(set(params.keys()) for params, _ in cluster_to_params[cluster_id])
            )
            total_weight = sum(weight for _, weight in cluster_to_params[cluster_id])
            if total_weight <= 0:
                total_weight = float(len(cluster_to_params[cluster_id]))

            averaged_params = {}
            param_names = [
                name for name, _ in model.named_parameters()
                if name in client_param_names
            ]
            for name in param_names:
                stacked = torch.stack(
                    [params[name].detach().to(self.device) * weight for params, weight in cluster_to_params[cluster_id]],
                    dim=0,
                )
                averaged_params[name] = stacked.sum(dim=0) / total_weight

            with torch.no_grad():
                for name, value in model.named_parameters():
                    if name not in averaged_params:
                        continue
                    mixed = (1.0 - self.model_mix) * value + self.model_mix * averaged_params[name].to(value.device)
                    value.copy_(mixed)
            sanitize_model_(model)

    def _should_recluster(self, round_idx):
        """向后兼容包装"""
        return self.should_recluster(round_idx + 1)

    def should_recluster(self, current_round):
        """放松版聚类策略：缩短 warmup，减少保守阶段，允许更频繁重聚类。"""
        if current_round < self.recluster_warmup_rounds:
            return False

        if self.recluster_every <= 0:
            return False

        if current_round % self.recluster_every != 0:
            return False

        if self.stop_recluster_after > 0 and current_round > self.stop_recluster_after:
            return False
        if self.max_reclusters > 0 and self.recluster_count >= self.max_reclusters:
            return False
        if self.skip_final_recluster and self.total_rounds is not None and current_round >= self.total_rounds:
            return False
        return True

    def should_apply_new_clustering(self, old_assignments, new_assignments):
        """放松版阈值：降低变动门槛，让有益的重分组更容易生效。"""
        if old_assignments is None or len(old_assignments) == 0:
            return True

        # 计算 client 变动比例
        changes = sum(1 for i in range(len(old_assignments))
                     if old_assignments[i] != new_assignments[i])
        change_ratio = changes / len(old_assignments)

        # 只有变动足够明显才应用新聚类
        if change_ratio > self.cluster_change_threshold:
            return True
        return False

    def store_clustering_snapshot(self, clients):
        """保存当前聚类分配用于对比"""
        return [client.cluster_id for client in clients]

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
            sanitize_model_(model)

    def recluster_clients(self, clients, client_cluster_vecs, old_assignments=None):
        """改进 3: 重聚类时检查变化, 避免微小波动导致的频繁重组"""
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

        # 改进 3: 检查新聚类是否应该应用
        if old_assignments is not None and not self.should_apply_new_clustering(old_assignments, assignments):
            return  # 不应用新聚类

        for client, cluster_id in zip(clients, assignments):
            client.cluster_id = int(cluster_id)
        self.recluster_count += 1

    def train_round(self, clients, round_idx, inner_lr=0.1, first_order=True, local_epochs=1):
        cluster_to_grads = defaultdict(list)
        cluster_to_params = defaultdict(list)
        client_cluster_vecs = []
        stats_list = []

        # 保存旧的聚类分配用于对比 (改进 3)
        old_cluster_assignment = self.store_clustering_snapshot(clients)

        for client in clients:
            # 改进 2: 动态调整内循环轮数
            dynamic_epochs = client.compute_dynamic_inner_epochs()

            model = self.cluster_models[client.cluster_id]
            try:
                meta_grads, update_vec, head_update_vec, adapted_params, stats = client.local_adapt_and_meta_grad(
                    model,
                    inner_lr=inner_lr,
                    first_order=first_order,
                    local_epochs=dynamic_epochs,  # ← 使用动态轮数
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MCFL train_round failed for client_id={client.client_id}, "
                    f"cluster_id={client.cluster_id}, round_idx={round_idx}"
                ) from exc

            client_weight = max(int(stats["query_samples"]), 1)
            cluster_to_grads[client.cluster_id].append((meta_grads, client_weight))
            cluster_to_params[client.cluster_id].append((adapted_params, client_weight))
            client_cluster_vecs.append(head_update_vec if self.cluster_feature == "head_updates" else update_vec)
            stats_list.append(stats)

        self.aggregate_meta_grads(cluster_to_grads)
        self._blend_cluster_models(cluster_to_params)

        if self.should_recluster(round_idx):
            # 改进 3: 传入旧的聚类分配用于对比
            self.recluster_clients(clients, client_cluster_vecs, old_assignments=old_cluster_assignment)

        return stats_list
