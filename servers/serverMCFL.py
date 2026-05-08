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
        algorithm="fomaml",
        enable_dynamic_clustering=False,
        outlier_threshold=0.3,
        drift_severity_low=0.5,
        drift_severity_high=0.2,
        outlier_pooling_threshold=3,
        agglomerative_threshold=0.5,
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
        self.algorithm = algorithm
        self.recluster_count = 0
        
        # 动态聚类参数
        self.enable_dynamic_clustering = enable_dynamic_clustering
        self.outlier_threshold = outlier_threshold
        self.drift_severity_low = drift_severity_low
        self.drift_severity_high = drift_severity_high
        self.outlier_pooling_threshold = outlier_pooling_threshold
        self.agglomerative_threshold = agglomerative_threshold
        self.outlier_pool = set()  # 存储孤立点的 client_id（去重）

        # 每轮动态聚类统计信息，供主循环打印日志
        self.last_dynamic_cluster_summary = {
            "outliers": 0,
            "new_clusters": 0,
            "total_clusters": self.num_clusters,
            "reassigned": 0,
        }

        # 用于检测剧烈漂移的 EMA 追踪
        self.loss_ema = None
        self.drift_threshold = 1.2  # 损失突增 20% 视为剧烈漂移
        self.ema_alpha = 0.1        # 平滑因子

        self.cluster_models = [
            copy.deepcopy(global_model).to(device)
            for _ in range(num_clusters)
        ]
        self.base_model = global_model  # 保存基础模型用于计算相似度

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

        # 动态支持所有存在的簇（即使簇数变化）
        max_cluster_id = max(cluster_to_params.keys()) if cluster_to_params else -1
        for cluster_id in range(max(self.num_clusters, max_cluster_id + 1)):
            if len(cluster_to_params.get(cluster_id, [])) == 0:
                continue

            # 如果簇 ID 超出当前 cluster_models 范围，跳过
            if cluster_id >= len(self.cluster_models):
                continue

            model = self.cluster_models[cluster_id]
            params_list = cluster_to_params[cluster_id]
            client_param_names = set.intersection(
                *(set(params.keys()) for params, _ in params_list)
            )
            total_weight = sum(weight for _, weight in params_list)
            if total_weight <= 0:
                total_weight = float(len(params_list))

            averaged_params = {}
            param_names = [
                name for name, _ in model.named_parameters()
                if name in client_param_names
            ]
            for name in param_names:
                stacked = torch.stack(
                    [params[name].detach().to(self.device) * weight for params, weight in params_list],
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

    def _build_cluster_centroids(self, clients, client_cluster_vecs):
        """根据当前客户端分配，构建每个簇在同一特征空间中的中心向量。"""
        cluster_to_vecs = defaultdict(list)
        for client, vec in zip(clients, client_cluster_vecs):
            cluster_to_vecs[int(client.cluster_id)].append(vec.detach().to(self.device))

        centroids = {}
        for cluster_id, vecs in cluster_to_vecs.items():
            if len(vecs) == 0:
                continue
            centroid = torch.stack(vecs, dim=0).mean(dim=0)
            centroid = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)
            centroids[cluster_id] = centroid
        return centroids

    def _compute_similarity_to_clusters(self, update_vec, cluster_centroids):
        """计算一个客户端与所有簇中心的相似度 (余弦相似度)"""
        update_vec = torch.as_tensor(update_vec, dtype=torch.float32, device=self.device)
        update_vec = F.normalize(update_vec.unsqueeze(0), dim=-1).squeeze(0)

        similarities = []
        for cluster_id in range(self.num_clusters):
            centroid = cluster_centroids.get(cluster_id)
            if centroid is None:
                similarities.append(-1.0)
                continue
            if centroid.shape != update_vec.shape:
                similarities.append(-1.0)
                continue
            sim = torch.nn.functional.cosine_similarity(update_vec.unsqueeze(0), centroid.unsqueeze(0)).item()
            similarities.append(sim)
        return similarities

    def _detect_outliers_and_drift(self, clients, client_cluster_vecs, stats_list):
        """
        孤立点检测与漂移分类：
        - 轻微漂移：与当前簇相似度低，但与某个其他簇相似度高 → 可跳槽
        - 中度漂移：与所有簇都有一定相似度，内循环可掩盖
        - 剧烈漂移：与所有簇都相似度很低 → 标记为孤立点
        """
        outliers = []
        drift_candidates = []  # (client_id, best_new_cluster_id)
        cluster_centroids = self._build_cluster_centroids(clients, client_cluster_vecs)

        for _, (client, vec, stats) in enumerate(zip(clients, client_cluster_vecs, stats_list)):
            sims = self._compute_similarity_to_clusters(vec, cluster_centroids)
            if len(sims) == 0:
                continue
            max_sim = float(max(sims))
            current_cluster_sim = -1.0
            current_cluster_id = int(client.cluster_id)
            for cid, sim in enumerate(sims):
                if cid == current_cluster_id:
                    current_cluster_sim = float(sim)
                    break

            if max_sim < self.outlier_threshold:
                # 剧烈漂移：与所有簇都不相像 → 候选孤立点
                outliers.append(client.client_id)
                print(f"[MCFL] Client {client.client_id} detected as outlier (max_sim={max_sim:.3f} < {self.outlier_threshold})")
            elif current_cluster_sim < self.drift_severity_low:
                # 轻微/中度漂移：看是否可从其他簇获益
                other_best_cluster = max(
                    range(len(sims)),
                    key=lambda j: float(sims[j]) if j != current_cluster_id else -1.0,
                )
                other_best_sim = float(sims[other_best_cluster])
                if other_best_sim > current_cluster_sim and other_best_sim > self.drift_severity_high:
                    drift_candidates.append((client.client_id, other_best_cluster))
                    print(f"[MCFL] Client {client.client_id} may benefit from switching to cluster {other_best_cluster} "
                          f"(current sim={current_cluster_sim:.3f}, best sim={other_best_sim:.3f})")

        return outliers, drift_candidates

    def _create_new_cluster(self, global_model=None):
        """创建新簇模型"""
        template_model = global_model if global_model is not None else self.base_model
        new_cluster_model = copy.deepcopy(template_model).to(self.device)
        new_cluster_models_list = self.cluster_models + [new_cluster_model]
        self.cluster_models = new_cluster_models_list
        self.num_clusters = len(self.cluster_models)
        print(f"[MCFL] New cluster created! Now have {self.num_clusters} clusters.")
        return self.num_clusters - 1  # 返回新簇的 ID

    def _handle_dynamic_clustering(self, clients, client_cluster_vecs, stats_list, global_model):
        """
        动态聚类处理流程：
        1. 检测孤立点和漂移
        2. 尝试将轻微漂移的客户端重新分配
        3. 当孤立点积累到阈值时，创建新簇
        """
        if not self.enable_dynamic_clustering:
            self.last_dynamic_cluster_summary = {
                "outliers": 0,
                "new_clusters": 0,
                "total_clusters": self.num_clusters,
                "reassigned": 0,
            }
            return

        outliers, drift_candidates = self._detect_outliers_and_drift(clients, client_cluster_vecs, stats_list)
        reassigned_count = 0

        # 处理轻微漂移：允许从其他簇跳槽
        for client_id, new_cluster_id in drift_candidates:
            for client in clients:
                if client.client_id == client_id:
                    old_cluster = client.cluster_id
                    client.cluster_id = new_cluster_id
                    reassigned_count += 1
                    print(f"[MCFL] Client {client_id} reassigned from cluster {old_cluster} to {new_cluster_id}")
                    break

        # 积累孤立点
        self.outlier_pool.update(outliers)

        # 如果孤立点达到阈值，创建新簇并分配它们
        new_clusters_created = 0
        if len(self.outlier_pool) >= self.outlier_pooling_threshold:
            new_cluster_id = self._create_new_cluster(global_model)
            new_clusters_created = 1
            for client_id in list(self.outlier_pool):
                for client in clients:
                    if client.client_id == client_id:
                        client.cluster_id = new_cluster_id
                        break
            self.outlier_pool.clear()  # 清空积累的孤立点

        self.last_dynamic_cluster_summary = {
            "outliers": len(outliers),
            "new_clusters": new_clusters_created,
            "total_clusters": self.num_clusters,
            "reassigned": reassigned_count,
        }


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
        # 动态支持所有存在的簇（即使簇数变化）
        max_cluster_id = max(cluster_to_grads.keys()) if cluster_to_grads else -1
        for cluster_id in range(max(self.num_clusters, max_cluster_id + 1)):
            if len(cluster_to_grads.get(cluster_id, [])) == 0:
                continue

            # 如果簇 ID 超出当前 cluster_models 范围，跳过
            if cluster_id >= len(self.cluster_models):
                continue

            model = self.cluster_models[cluster_id]
            grads = cluster_to_grads[cluster_id]
            num_params = len(list(model.parameters()))
            total_weight = sum(weight for _, weight in grads)
            if total_weight <= 0:
                total_weight = float(len(grads))

            avg_grads = []
            for p_idx in range(num_params):
                stacked = torch.stack(
                    [g[p_idx].detach() * weight for g, weight in grads],
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
                # 当轻微漂移发生时，这里的元适配 (inner loop) 能自动掩盖漂移
                meta_grads, update_vec, head_update_vec, adapted_params, stats = client.local_adapt_and_meta_grad(
                    model,
                    inner_lr=inner_lr,
                    first_order=first_order,
                    local_epochs=dynamic_epochs,  # ← 使用动态轮数
                    algorithm=self.algorithm,  # ← 传递算法参数
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

        # 剧烈漂移检测：基于平均 query_loss 突然升高
        avg_query_loss = sum(s["query_loss"] for s in stats_list) / max(len(stats_list), 1)
        severe_drift_detected = False
        
        if self.loss_ema is None:
            self.loss_ema = avg_query_loss
        else:
            # 当元适配无法再掩盖漂移时，损失会激增，触发剧烈漂移
            if avg_query_loss > self.drift_threshold * self.loss_ema:
                severe_drift_detected = True
                print(f"[MCFL] Severe drift detected! Loss spiked from {self.loss_ema:.4f} to {avg_query_loss:.4f}. Triggering re-clustering.")
            self.loss_ema = self.ema_alpha * avg_query_loss + (1 - self.ema_alpha) * self.loss_ema

        # 如果发生剧烈漂移，或者达到常规的重聚类周期，则触发低成本的簇重组
        if severe_drift_detected or self.should_recluster(round_idx):
            # 传入旧的聚类分配用于对比，以确保重组是低成本的（避免不必要的频繁小范围换簇）
            self.recluster_clients(clients, client_cluster_vecs, old_assignments=old_cluster_assignment)

        # 动态聚类处理：检测漂移并根据需要分裂或合并簇
        self._handle_dynamic_clustering(clients, client_cluster_vecs, stats_list, self.cluster_models[0] if self.cluster_models else None)

        return stats_list
