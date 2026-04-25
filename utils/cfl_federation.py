import random

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch import nn


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if hasattr(value, "data") and torch.is_tensor(value.data):
        return value.data.detach().clone()
    return torch.as_tensor(value).clone()


def copy_state(target, source):
    for name in target:
        target_tensor = target[name].data if hasattr(target[name], "data") else target[name]
        target_tensor.copy_(_clone_value(source[name]).to(target_tensor.device))


def subtract_state(target, minuend, subtrahend):
    for name in target:
        target_tensor = target[name].data if hasattr(target[name], "data") else target[name]
        target_tensor.copy_(
            _clone_value(minuend[name]).to(target_tensor.device)
            - _clone_value(subtrahend[name]).to(target_tensor.device)
        )


def reduce_add_average(targets, sources):
    if not targets or not sources:
        return

    for name in targets[0]:
        stacked = torch.stack([_clone_value(source[name]).to(targets[0][name].device) for source in sources], dim=0)
        averaged = torch.mean(stacked, dim=0)
        for target in targets:
            target_tensor = target[name].data if hasattr(target[name], "data") else target[name]
            target_tensor.add_(averaged.to(target_tensor.device))


def flatten(source):
    return torch.cat([_clone_value(value).flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()


def train_op(model, loader, optimizer, device, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss, samples = 0.0, 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

    return running_loss / samples if samples > 0 else 0.0


def eval_op(model, loader, device):
    model.eval()
    samples, correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct / samples if samples > 0 else 0.0


def pairwise_cluster_split(S):
    try:
        clustering = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete").fit(-S)
    except TypeError:
        clustering = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="complete").fit(-S)

    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    return c1, c2


def select_clients(clients, frac=1.0):
    if not clients:
        return []
    n_selected = max(1, int(len(clients) * frac))
    n_selected = min(len(clients), n_selected)
    return random.sample(list(clients), n_selected)

