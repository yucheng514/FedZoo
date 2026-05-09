import os

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from utils.data_utils import canonical_dataset_name, read_client_data


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
    dataset = canonical_dataset_name(dataset)
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
    all_train_ds = []
    for cid in range(args.num_clients):
        train_samples = read_client_data(args.dataset, cid, is_train=True, few_shot=args.few_shot)
        test_samples = read_client_data(args.dataset, cid, is_train=False, few_shot=args.few_shot)
        train_x, train_y = _stack_samples(train_samples, flatten=flatten)
        test_x, test_y = _stack_samples(test_samples, flatten=flatten)
        from torch.utils.data import TensorDataset
        all_train_ds.append(TensorDataset(train_x, train_y))
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
    # If heavy drift swap specification provided, build partner_map and attach DriftDataset wrappers
    swap_spec = getattr(args, 'drift_swap_clients', '')
    if swap_spec:
        # parse "a-b,c-d" or "a,c" style
        def expand_range(s):
            s = s.strip()
            if '-' in s:
                a, b = s.split('-', 1)
                return list(range(int(a), int(b) + 1))
            if ',' in s:
                return [int(p) for p in s.split(',') if p]
            return [int(s)]

        parts = [p.strip() for p in swap_spec.split(',', 1)]
        if len(parts) == 2 and '-' in parts[0] or '-' in parts[1] or parts[0].isdigit() or ',' in parts[0]:
            # allow forms like "0-4,5-9" or "0,1,2,3,4,5,6,7,8,9"
            # try to split into two groups by comma at top-level
            left, right = swap_spec.split(',', 1)
            group_a = expand_range(left)
            group_b = expand_range(right)
        else:
            # fallback: treat as two single ids separated by comma
            tokens = [t for t in swap_spec.split(',') if t]
            if len(tokens) >= 2:
                group_a = [int(tokens[0])]
                group_b = [int(tokens[1])]
            else:
                group_a = []
                group_b = []

        partner_map = {}
        # map each client in a-> corresponding b and b->a (cycle if sizes differ)
        if group_a and group_b:
            for i, cid in enumerate(sorted(group_a)):
                partner_map[int(cid)] = int(sorted(group_b)[i % len(group_b)])
            for i, cid in enumerate(sorted(group_b)):
                partner_map[int(cid)] = int(sorted(group_a)[i % len(group_a)])

            # attach DriftDataset wrappers to clients
            try:
                from utils.data_utils import DriftDataset
                for client in clients:
                    train_idx = client.client_id
                    ds = DriftDataset(
                        all_train_ds[train_idx],
                        client_id=client.client_id,
                        drift_type=getattr(args, 'drift_type', 'none'),
                        drift_every=getattr(args, 'drift_every', 5),
                        noise_step=getattr(args, 'drift_noise_step', 0.01),
                        noise_max=getattr(args, 'drift_noise_max', 0.10),
                        rotation_step=getattr(args, 'drift_rotation_step', 5.0),
                        heavy_round=getattr(args, 'drift_round', 25),
                        partner_map=partner_map,
                        all_client_data=all_train_ds,
                    )
                    client.drift_dataset = ds
            except Exception:
                pass
    return clients


def make_partitioned_cfl_data(args):
    client_data = []
    test_sets = []
    all_train_ds = []
    for cid in range(args.num_clients):
        train_samples = read_client_data(args.dataset, cid, is_train=True, few_shot=args.few_shot)
        test_samples = read_client_data(args.dataset, cid, is_train=False, few_shot=args.few_shot)
        train_x, train_y = _stack_samples(train_samples, flatten=False)
        test_x, test_y = _stack_samples(test_samples, flatten=False)
        from torch.utils.data import TensorDataset
        all_train_ds.append(TensorDataset(train_x, train_y))
        client_data.append(
            {
                "train": TensorDataset(train_x, train_y),
                "eval": TensorDataset(test_x, test_y),
            }
        )
        test_sets.append(TensorDataset(test_x, test_y))

    return client_data, ConcatDataset(test_sets)
