from dataclasses import dataclass

import torch
from torchvision import datasets

from dataset.download_paths import resolve_torchvision_root

@dataclass
class IFCATensorClient:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    cluster_id: int

    @property
    def train_samples(self):
        return int(self.train_x.shape[0])

    @property
    def test_samples(self):
        return int(self.test_x.shape[0])


def _rotation_k(num_clusters, cluster_id):
    if num_clusters == 1:
        return 0
    if num_clusters == 2:
        return (cluster_id % 2) * 2
    if num_clusters == 4:
        return cluster_id
    return cluster_id % 4


def _chunk_indices(num_points, num_clients):
    if num_points % num_clients != 0:
        raise ValueError(f"Dataset of size {num_points} is not evenly divisible by {num_clients} clients.")
    chunk_size = num_points // num_clients
    return [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(num_clients)]


def _build_split_clients(images, labels, num_clients, num_clusters):
    if num_clients % num_clusters != 0:
        raise ValueError("IFCA CIFAR setup requires num_clients divisible by ifca_clusters.")

    clients = []
    clients_per_cluster = num_clients // num_clusters
    for client_idx, data_slice in enumerate(_chunk_indices(len(labels), num_clients)):
        cluster_id = client_idx // clients_per_cluster
        x = images[data_slice]
        y = labels[data_slice]
        x = torch.rot90(x, k=_rotation_k(num_clusters, cluster_id), dims=(2, 3)).float()
        clients.append((x, y.long(), cluster_id))
    return clients


def make_ifca_rotated_cifar_clients(args):
    data_root = resolve_torchvision_root(args.ifca_data_root, "CIFAR10")
    train_set = datasets.CIFAR10(root=str(data_root), train=True, download=args.ifca_download)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=args.ifca_download)

    train_images = torch.tensor(train_set.data).permute(0, 3, 1, 2).float() / 255.0
    train_labels = torch.tensor(train_set.targets)
    test_images = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
    test_labels = torch.tensor(test_set.targets)

    train_clients = _build_split_clients(train_images, train_labels, args.num_clients, args.ifca_clusters)
    test_clients = _build_split_clients(test_images, test_labels, args.num_clients, args.ifca_clusters)

    clients = []
    for train_info, test_info in zip(train_clients, test_clients):
        train_x, train_y, cluster_id = train_info
        test_x, test_y, _ = test_info
        clients.append(
            IFCATensorClient(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                cluster_id=cluster_id,
            )
        )

    return clients
