import copy

import torch


class IFCAServer:
    def __init__(self, cluster_models, clients, criterion, task, device, mode="clustered", freeze_backbone=False):
        self.cluster_models = [model.to(device) for model in cluster_models]
        self.clients = clients
        self.criterion = criterion
        self.task = task
        self.device = device
        self.mode = mode
        self.freeze_backbone = freeze_backbone
        self.fixed_assignments = None

    def warmstart_clusters(self, assignments, lr, local_epochs, rounds):
        if rounds <= 0:
            return
        for _ in range(rounds):
            grouped_updates = [[] for _ in self.cluster_models]
            grouped_weights = [[] for _ in self.cluster_models]
            for client, cluster_idx in zip(self.clients, assignments):
                update = client.local_update(
                    base_model=self.cluster_models[cluster_idx],
                    criterion=self.criterion,
                    lr=lr,
                    local_epochs=local_epochs,
                    freeze_backbone=self.freeze_backbone,
                )
                grouped_updates[cluster_idx].append(update["model"])
                grouped_weights[cluster_idx].append(client.train_samples)
            for cluster_idx, local_models in enumerate(grouped_updates):
                if local_models:
                    self.cluster_models[cluster_idx] = self._aggregate(local_models, grouped_weights[cluster_idx]).to(self.device)

    def initialize_fixed_assignments(self, strategy="random"):
        if strategy == "random":
            self.fixed_assignments = [client.id % len(self.cluster_models) for client in self.clients]
        elif strategy == "loss":
            self.fixed_assignments = self.assign_clients()
        else:
            raise ValueError(f"Unsupported fixed assignment strategy: {strategy}")

    def assign_clients(self):
        if self.mode == "local":
            return [client.id for client in self.clients]
        if self.mode == "oneshot" and self.fixed_assignments is not None:
            return list(self.fixed_assignments)
        assignments = []
        for client in self.clients:
            losses = [client.loss_for_model(model, self.criterion, train=True) for model in self.cluster_models]
            assignments.append(min(range(len(losses)), key=lambda idx: losses[idx]))
        return assignments

    def train_round(self, lr, local_epochs):
        assignments = self.assign_clients()
        grouped_updates = [[] for _ in self.cluster_models]
        grouped_weights = [[] for _ in self.cluster_models]
        train_losses = []

        for client, cluster_idx in zip(self.clients, assignments):
            update = client.local_update(
                base_model=self.cluster_models[cluster_idx],
                criterion=self.criterion,
                lr=lr,
                local_epochs=local_epochs,
                freeze_backbone=self.freeze_backbone,
            )
            grouped_updates[cluster_idx].append(update["model"])
            grouped_weights[cluster_idx].append(client.train_samples)
            train_losses.append(update["train_loss"])

        for cluster_idx, local_models in enumerate(grouped_updates):
            if local_models:
                self.cluster_models[cluster_idx] = self._aggregate(local_models, grouped_weights[cluster_idx]).to(self.device)

        return {
            "assignments": assignments,
            "train_loss": float(sum(train_losses) / len(train_losses)) if train_losses else 0.0,
        }

    def evaluate(self):
        assignments = self.assign_clients()
        train_losses = []
        test_scores = []
        cluster_acc = []
        has_cluster_labels = False

        for client, cluster_idx in zip(self.clients, assignments):
            model = self.cluster_models[cluster_idx]
            train_losses.append(client.loss_for_model(model, self.criterion, train=True))
            test_metric = client.metric_for_model(model, train=False)
            test_scores.append(test_metric)
            if getattr(client, "cluster_id", -1) >= 0:
                has_cluster_labels = True
                cluster_acc.append(float(cluster_idx == client.cluster_id))

        assignment_hist = {}
        for cluster_idx in assignments:
            assignment_hist[cluster_idx] = assignment_hist.get(cluster_idx, 0) + 1

        result = {
            "train_loss": float(sum(train_losses) / len(train_losses)) if train_losses else 0.0,
            "cluster_acc": float(sum(cluster_acc) / len(cluster_acc)) if has_cluster_labels and cluster_acc else -1.0,
            "assignment_hist": assignment_hist,
        }

        if self.task == "regression":
            values = [score["mse"] for score in test_scores]
            result["test_mse"] = float(sum(values) / len(values)) if values else 0.0
        else:
            values = [score["acc"] for score in test_scores]
            result["test_acc"] = float(sum(values) / len(values)) if values else 0.0

        return result

    def _aggregate(self, local_models, weights):
        global_model = copy.deepcopy(local_models[0])
        total_weight = float(sum(weights))

        state_dict = global_model.state_dict()
        for name in state_dict:
            state_dict[name] = torch.zeros_like(state_dict[name])

        for weight, model in zip(weights, local_models):
            model_state = model.state_dict()
            for name in state_dict:
                state_dict[name] += model_state[name] * (weight / total_weight)

        global_model.load_state_dict(state_dict)
        return global_model
