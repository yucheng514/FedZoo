"""IFCA support for pre-partitioned CIFAR10 data (from federated dataset generation)."""

from dataclasses import dataclass

import torch

from utils.data_utils import read_client_data


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


def _stack_samples(samples, flatten=False):
    xs, ys = [], []
    for x, y in samples:
        x = x.float()
        if flatten:
            x = x.reshape(-1)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        xs.append(x)
        ys.append(int(y.item()) if torch.is_tensor(y) else int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def make_ifca_partitioned_cifar_clients(args):
    """Create IFCA clients from pre-partitioned CIFAR10 data."""
    clients = []
    for cid in range(args.num_clients):
        train_samples = read_client_data(args.dataset, cid, is_train=True, few_shot=args.few_shot)
        test_samples = read_client_data(args.dataset, cid, is_train=False, few_shot=args.few_shot)
        train_x, train_y = _stack_samples(train_samples, flatten=False)
        test_x, test_y = _stack_samples(test_samples, flatten=False)
        
        # cluster_id=-1 means unknown (no true cluster labels in real data)
        # IFCA will learn cluster assignments from data during training
        cluster_id = -1
        
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

