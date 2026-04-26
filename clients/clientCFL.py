import torch
from torch.utils.data import DataLoader, random_split

from utils.cfl_federation import copy_state, subtract_state, train_op, eval_op


class CFLClient:
    def __init__(self, args, idnum, data, model_fn, optimizer_fn, batch_size=128, train_frac=0.8, seed=0):
        torch.manual_seed(seed + idnum)
        self.args = args
        self.device = args.device
        self.id = idnum
        self.data = data
        self.model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        if isinstance(data, dict) and "train" in data and "eval" in data:
            data_train = data["train"]
            data_eval = data["eval"]
        else:
            n_total = len(data)
            if n_total < 2:
                raise ValueError(f"Client {idnum} needs at least 2 samples, got {n_total}.")

            n_train = max(1, int(round(n_total * train_frac)))
            n_train = min(n_train, n_total - 1)
            n_eval = n_total - n_train
            generator = torch.Generator().manual_seed(seed + idnum)
            data_train, data_eval = random_split(self.data, [n_train, n_eval], generator=generator)

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False, drop_last=False)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

    def synchronize_with_server(self, server):
        copy_state(target=self.W, source=server.W)

    def compute_weight_update(self, epochs=1, loader=None):
        copy_state(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"] *= 0.99
        train_stats = train_op(self.model, self.train_loader if loader is None else loader, self.optimizer, self.device, epochs)
        subtract_state(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats

    def reset(self):
        copy_state(target=self.W, source=self.W_old)

    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if loader is None else loader, self.device)
