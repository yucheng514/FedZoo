import torch
import torch.nn.functional as F

from torch.func import functional_call

from utils.mcfl_utils import clone_params_dict, infer_head_param_names, vectorize_params_dict, vectorize_selected_params


class MCFLClient:
    def __init__(self, client_id, support_loader, query_loader, test_loader=None, device="cpu", local_epochs=1):
        self.client_id = client_id
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.test_loader = test_loader if test_loader is not None else query_loader
        self.device = device
        self.local_epochs = local_epochs
        self.cluster_id = 0

    def _adapt_params(self, meta_model, inner_lr=0.1, local_epochs=None):
        meta_model = meta_model.to(self.device)
        meta_model.train()

        if local_epochs is None:
            local_epochs = self.local_epochs

        params = clone_params_dict(meta_model)
        for _ in range(local_epochs):
            for x_s, y_s in self.support_loader:
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)

                logits_s = functional_call(meta_model, params, (x_s,))
                support_loss = F.cross_entropy(logits_s, y_s)
                support_grads = torch.autograd.grad(
                    support_loss,
                    list(params.values()),
                    create_graph=False,
                    allow_unused=False,
                )
                params = {
                    name: p - inner_lr * g
                    for (name, p), g in zip(params.items(), support_grads)
                }

        return params

    def local_adapt_and_meta_grad(self, meta_model, inner_lr=0.1, first_order=True, local_epochs=None):
        meta_model = meta_model.to(self.device)
        meta_model.train()

        if local_epochs is None:
            local_epochs = self.local_epochs

        params = clone_params_dict(meta_model)
        original_params = {name: p for name, p in params.items()}
        head_param_names = infer_head_param_names(params)

        def _check_labels(logits, targets, phase):
            if targets.numel() == 0:
                raise ValueError(f"MCFL {phase} batch is empty.")
            if targets.dtype != torch.long:
                raise TypeError(f"MCFL {phase} labels must be torch.long, got {targets.dtype}.")
            num_classes = logits.shape[-1]
            min_label = int(targets.min().item())
            max_label = int(targets.max().item())
            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"MCFL {phase} labels out of range: min={min_label}, max={max_label}, num_classes={num_classes}."
                )

        support_loss_total = 0.0
        support_samples = 0
        support_correct = 0
        for _ in range(local_epochs):
            for x_s, y_s in self.support_loader:
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)

                logits_s = functional_call(meta_model, params, (x_s,))
                _check_labels(logits_s, y_s, "support")
                support_loss = F.cross_entropy(logits_s, y_s)

                support_grads = torch.autograd.grad(
                    support_loss,
                    list(params.values()),
                    create_graph=not first_order,
                    allow_unused=False,
                )

                params = {
                    name: p - inner_lr * g
                    for (name, p), g in zip(params.items(), support_grads)
                }

                preds_s = torch.argmax(logits_s, dim=1)
                support_correct += (preds_s == y_s).sum().item()
                support_loss_total += support_loss.item() * y_s.size(0)
                support_samples += y_s.size(0)

        query_loss_total = 0.0
        query_samples = 0
        query_correct = 0
        query_loss_tensor = None
        for x_q, y_q in self.query_loader:
            x_q = x_q.to(self.device)
            y_q = y_q.to(self.device)
            logits_q = functional_call(meta_model, params, (x_q,))
            _check_labels(logits_q, y_q, "query")
            batch_query_loss = F.cross_entropy(logits_q, y_q)

            weighted_loss = batch_query_loss * y_q.size(0)
            query_loss_tensor = weighted_loss if query_loss_tensor is None else query_loss_tensor + weighted_loss
            preds_q = torch.argmax(logits_q, dim=1)
            query_correct += (preds_q == y_q).sum().item()
            query_loss_total += weighted_loss.item()
            query_samples += y_q.size(0)

        if query_loss_tensor is None:
            raise RuntimeError("MCFL query loader is empty.")

        query_loss = query_loss_tensor / query_samples

        meta_grads = torch.autograd.grad(
            query_loss,
            list(original_params.values()),
            create_graph=False,
            allow_unused=False,
        )

        original_vec = vectorize_params_dict(original_params).detach()
        adapted_vec = vectorize_params_dict(params).detach()
        update_vec = adapted_vec - original_vec
        head_update_vec = (
            vectorize_selected_params(params, head_param_names).detach()
            - vectorize_selected_params(original_params, head_param_names).detach()
        )
        adapted_params = {
            name: value.detach().clone()
            for name, value in params.items()
        }

        stats = {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "support_loss": support_loss_total / max(support_samples, 1),
            "query_loss": query_loss_total / max(query_samples, 1),
            "support_acc": support_correct / max(support_samples, 1),
            "query_acc": query_correct / max(query_samples, 1),
            "support_samples": support_samples,
            "query_samples": query_samples,
            "support_correct": support_correct,
            "query_correct": query_correct,
        }

        return meta_grads, update_vec, head_update_vec, adapted_params, stats

    def evaluate(self, model, adapt=False, inner_lr=0.1, local_epochs=None):
        model = model.to(self.device)
        eval_params = None
        if adapt:
            eval_params = self._adapt_params(model, inner_lr=inner_lr, local_epochs=local_epochs)

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                if eval_params is None:
                    logits = model(x)
                else:
                    logits = functional_call(model, eval_params, (x,))
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)
