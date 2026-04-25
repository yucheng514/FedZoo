import torch
import torch.nn.functional as F

from torch.func import functional_call

from utils.mcfl_utils import clone_params_dict, vectorize_params_dict


class MCFLClient:
    def __init__(self, client_id, support_loader, query_loader, device="cpu", local_epochs=1):
        self.client_id = client_id
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.device = device
        self.local_epochs = local_epochs
        self.cluster_id = 0

    def local_adapt_and_meta_grad(self, meta_model, inner_lr=0.1, first_order=True, local_epochs=None):
        meta_model = meta_model.to(self.device)
        meta_model.train()

        if local_epochs is None:
            local_epochs = self.local_epochs

        params = clone_params_dict(meta_model)
        original_params = {name: p for name, p in params.items()}

        support_loss_total = 0.0
        support_samples = 0
        for _ in range(local_epochs):
            for x_s, y_s in self.support_loader:
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)

                logits_s = functional_call(meta_model, params, (x_s,))
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

                support_loss_total += support_loss.item() * y_s.size(0)
                support_samples += y_s.size(0)

        query_loss_total = 0.0
        query_samples = 0
        query_loss_tensor = None
        for x_q, y_q in self.query_loader:
            x_q = x_q.to(self.device)
            y_q = y_q.to(self.device)
            logits_q = functional_call(meta_model, params, (x_q,))
            batch_query_loss = F.cross_entropy(logits_q, y_q)

            weighted_loss = batch_query_loss * y_q.size(0)
            query_loss_tensor = weighted_loss if query_loss_tensor is None else query_loss_tensor + weighted_loss
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

        stats = {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "support_loss": support_loss_total / max(support_samples, 1),
            "query_loss": query_loss_total / max(query_samples, 1),
            "support_samples": support_samples,
            "query_samples": query_samples,
        }

        return meta_grads, update_vec, stats

