import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

from utils.mcfl_utils import get_one_batch, clone_params_dict, vectorize_params_dict


class MCFLClient:
    def __init__(self, client_id, support_loader, query_loader, device="cpu"):
        self.client_id = client_id
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.device = device
        self.cluster_id = 0

    def local_adapt_and_meta_grad(self, meta_model, inner_lr=0.1, first_order=True):
        meta_model = meta_model.to(self.device)
        x_s, y_s = get_one_batch(self.support_loader, self.device)
        x_q, y_q = get_one_batch(self.query_loader, self.device)

        params = clone_params_dict(meta_model)

        logits_s = functional_call(meta_model, params, (x_s,))
        support_loss = F.cross_entropy(logits_s, y_s)

        support_grads = torch.autograd.grad(
            support_loss,
            list(params.values()),
            create_graph=not first_order,
            allow_unused=False,
        )

        adapted_params = {
            name: p - inner_lr * g
            for (name, p), g in zip(params.items(), support_grads)
        }

        logits_q = functional_call(meta_model, adapted_params, (x_q,))
        query_loss = F.cross_entropy(logits_q, y_q)

        meta_grads = torch.autograd.grad(
            query_loss,
            list(params.values()),
            create_graph=False,
            allow_unused=False,
        )

        original_vec = vectorize_params_dict(params).detach()
        adapted_vec = vectorize_params_dict(adapted_params).detach()
        update_vec = adapted_vec - original_vec

        stats = {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "support_loss": support_loss.item(),
            "query_loss": query_loss.item(),
        }

        return meta_grads, update_vec, stats

