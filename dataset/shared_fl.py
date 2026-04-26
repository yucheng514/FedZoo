import os

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from utils.data_utils import read_client_data


class SharedTensorClient:
    def __init__(self, train_x, train_y, test_x, test_y, client_id=0, cluster_id=-1):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.client_id = client_id
        self.cluster_id = cluster_id

    @property
    def train_samples(self):
        return int(self.train_x.shape[0])

    @property
    def test_samples(self):
        return int(self.test_x.shape[0])


def has_partitioned_data(dataset):
    train_dir = os.path.join("./dataset/data", dataset, "train")
    test_dir = os.path.join("./dataset/data", dataset, "test")
    return os.path.isdir(train_dir) and os.path.isdir(test_dir)


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


def make_partitioned_tensor_clients(args, flatten=False):
    clients = []
    for cid in range(args.num_clients):
        train_samples = read_client_data(args.dataset, cid, is_train=True, few_shot=args.few_shot)
        test_samples = read_client_data(args.dataset, cid, is_train=False, few_shot=args.few_shot)
        train_x, train_y = _stack_samples(train_samples, flatten=flatten)
        test_x, test_y = _stack_samples(test_samples, flatten=flatten)
        clients.append(
            SharedTensorClient(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                client_id=cid,
                cluster_id=-1,
            )
        )
    return clients


def make_partitioned_cfl_data(args):
    client_data = []
    test_sets = []
    for cid in range(args.num_clients):
        train_samples = read_client_data(args.dataset, cid, is_train=True, few_shot=args.few_shot)
        test_samples = read_client_data(args.dataset, cid, is_train=False, few_shot=args.few_shot)
        train_x, train_y = _stack_samples(train_samples, flatten=False)
        test_x, test_y = _stack_samples(test_samples, flatten=False)
        client_data.append(
            {
                "train": TensorDataset(train_x, train_y),
                "eval": TensorDataset(test_x, test_y),
            }
        )
        test_sets.append(TensorDataset(test_x, test_y))

    return client_data, ConcatDataset(test_sets)
