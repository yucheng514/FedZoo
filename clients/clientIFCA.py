import copy

import torch


class IFCAClient:
    def __init__(self, client_id, data, task, device):
        self.id = client_id
        self.data = data
        self.task = task
        self.device = device

    @property
    def cluster_id(self):
        return self.data.cluster_id

    @property
    def train_samples(self):
        return self.data.train_samples

    @property
    def test_samples(self):
        return self.data.test_samples

    def _split(self, train=True, device=None):
        target_device = self.device if device is None else device
        if train:
            return self.data.train_x.contiguous().to(target_device), self.data.train_y.contiguous().to(target_device)
        return self.data.test_x.contiguous().to(target_device), self.data.test_y.contiguous().to(target_device)

    def _sanitize_model_(self, model, clamp=1e4):
        with torch.no_grad():
            for param in model.parameters():
                cleaned = torch.nan_to_num(param, nan=0.0, posinf=clamp, neginf=-clamp)
                param.copy_(torch.clamp(cleaned, min=-clamp, max=clamp))

    def clone_model_for_local(self, base_model, freeze_backbone=False):
        model = copy.deepcopy(base_model).to(self.device)
        if freeze_backbone:
            classifier_names = ("classifier", "fc", "head", "linear")
            for name, param in model.named_parameters():
                keep_trainable = any(token in name for token in classifier_names)
                param.requires_grad = keep_trainable
        return model

    def loss_for_model(self, model, criterion, train=True, device=None):
        target_device = self.device if device is None else device
        x, y = self._split(train=train, device=target_device)
        model = model.to(target_device)
        model.eval()
        with torch.no_grad():
            output = model(x)
            if not torch.isfinite(output).all():
                raise ValueError(f"IFCA model output contains non-finite values for client {self.id}.")
            loss = criterion(output, y)
            if not torch.isfinite(loss):
                raise ValueError(f"IFCA loss became non-finite for client {self.id}.")
        return float(loss.item())

    def metric_for_model(self, model, train=False, device=None):
        target_device = self.device if device is None else device
        x, y = self._split(train=train, device=target_device)
        model = model.to(target_device)
        model.eval()
        with torch.no_grad():
            output = model(x)
            if not torch.isfinite(output).all():
                raise ValueError(f"IFCA metric output contains non-finite values for client {self.id}.")
            if self.task == "regression":
                mse = torch.mean((output - y) ** 2).item()
                return {"mse": float(mse)}
            pred = torch.argmax(output, dim=1)
            acc = (pred == y).float().mean().item()
            return {"acc": float(acc)}

    def local_update(self, base_model, criterion, lr, local_epochs, freeze_backbone=False):
        model = self.clone_model_for_local(base_model, freeze_backbone=freeze_backbone)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=lr) if trainable_params else None
        x, y = self._split(train=True, device=self.device)

        model.train()
        if optimizer is not None:
            for _ in range(local_epochs):
                output = model(x)
                loss = criterion(output, y)
                if not torch.isfinite(output).all():
                    raise ValueError(f"IFCA local_update output contains non-finite values for client {self.id}.")
                if not torch.isfinite(loss):
                    raise ValueError(f"IFCA local_update loss became non-finite for client {self.id}.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._sanitize_model_(model)

        return {
            "model": copy.deepcopy(model).cpu(),
            "train_loss": self.loss_for_model(model, criterion, train=True, device=self.device),
            "test_metric": self.metric_for_model(model, train=False, device=self.device),
        }
