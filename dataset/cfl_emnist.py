from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from dataset.download_paths import resolve_torchvision_root
from dataset.shared_fl import has_partitioned_data, make_partitioned_cfl_data
from utils.cfl_data_utils import CustomSubset, split_noniid


def _client_transform(client_id, rotation_clients, rotation_degrees):
    if client_id < rotation_clients:
        return transforms.Compose([transforms.RandomRotation((rotation_degrees, rotation_degrees)), transforms.ToTensor()])
    return transforms.Compose([transforms.ToTensor()])


def make_cfl_partition(args):
    if has_partitioned_data(args.dataset):
        client_data, test_data = make_partitioned_cfl_data(args)
        return client_data, test_data, None

    data_root = resolve_torchvision_root(args.cfl_data_root, "EMNIST")
    data = datasets.EMNIST(root=str(data_root), split=args.cfl_split, download=args.cfl_download)

    idcs = np.random.permutation(len(data))
    train_size = min(args.cfl_train_samples, max(len(data) - 1, 1))
    test_size = min(args.cfl_test_samples, max(len(data) - train_size, 1))
    train_idcs = idcs[:train_size]
    test_idcs = idcs[train_size:train_size + test_size]

    train_labels = np.asarray(data.targets)
    client_idcs = split_noniid(train_idcs, train_labels, alpha=args.cfl_dirichlet_alpha, n_clients=args.num_clients)

    client_data = []
    rotation_clients = min(args.cfl_rotation_clients, len(client_idcs))
    for cid, idxs in enumerate(client_idcs):
        if len(idxs) == 0:
            raise ValueError(
                f"CFL partition produced an empty client {cid}. Try increasing cfl_train_samples or lowering num_clients."
            )
        subset = CustomSubset(data, idxs)
        subset.subset_transform = _client_transform(cid, rotation_clients, args.cfl_rotation_degrees)
        client_data.append(subset)

    test_data = CustomSubset(data, test_idcs, transforms.Compose([transforms.ToTensor()]))
    return client_data, test_data, client_idcs
