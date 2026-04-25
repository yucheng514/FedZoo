import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def get_one_batch(loader, device):
    x, y = next(iter(loader))
    return x.to(device), y.to(device)



def vectorize_params_dict(params_dict):
    return torch.cat([v.reshape(-1) for _, v in params_dict.items()])



def clone_params_dict(model):
    return {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

