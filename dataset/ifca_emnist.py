from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from dataset.download_paths import resolve_torchvision_root

class IFCATensorClient:
    def __init__(self, train_x, train_y, test_x, test_y, cluster_id):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cluster_id = cluster_id

    @property
    def train_samples(self):
        return int(self.train_x.shape[0])

    @property
    def test_samples(self):
        return int(self.test_x.shape[0])


def _dirichlet_partition(labels, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    min_size = 0
    while min_size == 0:
        client_indices = [[] for _ in range(num_clients)]
        for label in range(num_classes):
            idxs = np.where(labels == label)[0]
            rng.shuffle(idxs)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cut_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(idxs, cut_points)):
                client_indices[client_id].extend(split.tolist())
        min_size = min(len(idxs) for idxs in client_indices)
    return [np.asarray(sorted(idxs)) for idxs in client_indices]


def make_ifca_emnist_clients(args):
    data_root = resolve_torchvision_root(args.ifca_data_root, "EMNIST")
    train_set = datasets.EMNIST(root=str(data_root), split=args.ifca_emnist_split, train=True, download=args.ifca_download)
    test_set = datasets.EMNIST(root=str(data_root), split=args.ifca_emnist_split, train=False, download=args.ifca_download)

    train_images = train_set.data.unsqueeze(1).float() / 255.0
    train_labels = train_set.targets.numpy()
    test_images = test_set.data.unsqueeze(1).float() / 255.0
    test_labels = test_set.targets.numpy()

    train_partitions = _dirichlet_partition(train_labels, args.num_clients, args.ifca_dirichlet_alpha, args.ifca_seed)
    test_partitions = _dirichlet_partition(test_labels, args.num_clients, args.ifca_dirichlet_alpha, args.ifca_seed + 1)

    clients = []
    for client_id, (train_idx, test_idx) in enumerate(zip(train_partitions, test_partitions)):
        cluster_id = client_id % args.ifca_clusters
        clients.append(
            IFCATensorClient(
                train_x=train_images[train_idx],
                train_y=torch.tensor(train_labels[train_idx]).long(),
                test_x=test_images[test_idx],
                test_y=torch.tensor(test_labels[test_idx]).long(),
                cluster_id=cluster_id,
            )
        )
    return clients
