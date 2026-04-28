import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    """Per-FedAvg 优化器"""

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta != 0:
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group["lr"])


class pFedMeOptimizer(Optimizer):
    """pFedMe 优化器"""

    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """标准 step() 接口，兼容 torch.optim API"""
        return None

    def step_pfedme(self, local_model, device):
        """pFedMe 特定的参数更新"""
        weight_update = local_model.copy()
        group = None
        for group in self.param_groups:
            for p, localweight in zip(group["params"], weight_update):
                localweight = localweight.to(device)
                p.data = p.data - group["lr"] * (
                    p.grad.data + group["lamda"] * (p.data - localweight.data) + group["mu"] * p.data
                )

        return group["params"]

