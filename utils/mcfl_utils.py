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



def infer_head_param_names(params_dict):
    names = list(params_dict.keys())
    preferred = [
        name for name in names
        if any(token in name.lower() for token in ("classifier", "head", ".fc", "fc.", "fc", "output"))
    ]
    if preferred:
        return preferred

    if len(names) >= 2:
        return names[-2:]
    return names



def vectorize_selected_params(params_dict, selected_names):
    selected = [params_dict[name].reshape(-1) for name in selected_names if name in params_dict]
    if not selected:
        return vectorize_params_dict(params_dict)
    return torch.cat(selected)



def clone_params_dict(model):
    return {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
