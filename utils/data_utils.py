import numpy as np
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset


_DATASET_ALIASES = {
    "CIFAR10": "Cifar10",
    "CIFAR-10": "Cifar10",
}


def canonical_dataset_name(dataset):
    dataset = str(dataset)
    return _DATASET_ALIASES.get(dataset.upper(), dataset)


def read_data(dataset, idx, is_train=True):
    dataset = canonical_dataset_name(dataset)
    if is_train:
        data_dir = os.path.join('./dataset/data', dataset, 'train/')
    else:
        data_dir = os.path.join('./dataset/data', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


# Global drift control (set from main loop)
GLOBAL_DRIFT_ROUND = 0


def set_global_drift_round(r):
    global GLOBAL_DRIFT_ROUND
    GLOBAL_DRIFT_ROUND = int(r)


def build_partner_map_from_swap_spec(swap_spec):
    """Parse `drift_swap_clients` into a symmetric partner map.

    Supported formats:
      - "0-4,5-9"  -> 0..4 <-> 5..9 (cyclic if sizes differ)
      - "0,1"      -> client 0 <-> client 1
      - "0-3,8,9"  -> left group + right group tokens supported

    Returns an empty dict when the spec is empty or invalid.
    """
    swap_spec = str(swap_spec or "").strip()
    if not swap_spec:
        return {}

    def expand_group(group_text):
        group_text = group_text.strip()
        if not group_text:
            return []
        if '-' in group_text and group_text.count(',') == 0:
            start, end = group_text.split('-', 1)
            start, end = int(start), int(end)
            if start <= end:
                return list(range(start, end + 1))
            return list(range(start, end - 1, -1))
        return [int(token.strip()) for token in group_text.split(',') if token.strip()]

    try:
        left_text, right_text = swap_spec.split(',', 1)
        group_a = expand_group(left_text)
        group_b = expand_group(right_text)
    except Exception:
        return {}

    if not group_a or not group_b:
        return {}

    partner_map = {}
    group_a = sorted(set(int(x) for x in group_a))
    group_b = sorted(set(int(x) for x in group_b))
    for i, cid in enumerate(group_a):
        partner_map[int(cid)] = int(group_b[i % len(group_b)])
    for i, cid in enumerate(group_b):
        partner_map[int(cid)] = int(group_a[i % len(group_a)])
    return partner_map


class DriftDataset(Dataset):
    """Wraps a dataset (list of (x,y) or TensorDataset) and applies temporal drift.

    Mild drift: progressively add Gaussian noise or rotate images based on GLOBAL_DRIFT_ROUND.
    Heavy drift: toggle swapping samples from a partner client dataset based on drift_interval.
    """

    def __init__(
        self,
        data,
        client_id=0,
        drift_type='none',
        drift_every=5,
        noise_step=0.01,
        noise_max=0.10,
        rotation_step=5.0,
        drift_interval=25,
        partner_map=None,
        all_client_data=None,
    ):
        # data: list[(x,y)] or TensorDataset
        self.client_id = int(client_id)
        self._is_tensor_ds = hasattr(data, '__getitem__') and not isinstance(data, list)
        self.data = data
        self.drift_type = str(drift_type)
        self.drift_every = max(1, int(drift_every))
        self.noise_step = float(noise_step)
        self.noise_max = float(noise_max)
        self.rotation_step = float(rotation_step)
        self.drift_interval = int(drift_interval)
        # partner_map: dict mapping client_id -> partner_client_id for swapping samples
        self.partner_map = dict(partner_map) if partner_map else {}
        # all_client_data: list of original datasets for swapping
        self.all_client_data = all_client_data

    def __len__(self):
        if self._is_tensor_ds:
            return len(self.data)
        return len(self.data)

    def _get_raw_item(self, idx):
        if self._is_tensor_ds:
            return self.data[idx]
        return self.data[idx]

    def _apply_mild(self, x):
        # Only apply to image-like tensors (C,H,W) or (N,C,H,W) handled at batch level
        drift_step = GLOBAL_DRIFT_ROUND // self.drift_every
        if drift_step <= 0:
            return x
        out = x
        if self.noise_step > 0.0:
            noise_std = min(self.noise_max, drift_step * self.noise_step)
            if noise_std > 0.0:
                out = out + noise_std * torch.randn_like(out)
        if abs(self.rotation_step) > 0.0:
            angle = drift_step * self.rotation_step
            if abs(angle) > 0.0:
                try:
                    from torchvision.transforms.functional import rotate
                    out = rotate(out, angle=angle)
                except Exception:
                    pass
        return out

    def _get_from_partner(self, idx):
        # If partner_map and all_client_data are provided, fetch same-index sample from partner dataset
        if not self.partner_map or not self.all_client_data:
            return None
        partner_id = self.partner_map.get(self.client_id)
        if partner_id is None or partner_id < 0 or partner_id >= len(self.all_client_data):
            return None
        partner_ds = self.all_client_data[partner_id]
        if idx >= len(partner_ds):
            idx = idx % len(partner_ds)
        try:
            return partner_ds[idx]
        except Exception:
            return None

    def __getitem__(self, idx):
        raw = self._get_raw_item(idx)
        if isinstance(raw, tuple) and len(raw) == 2:
            x, y = raw
        else:
            # TensorDataset returns a tuple
            x, y = raw[0], raw[1]

        # Heavy drift: toggle swap with partner dataset every drift_interval rounds
        if self.drift_type in ('heavy', 'both') and self.drift_interval > 0:
            # e.g., if drift_interval=50: rounds 50-99 swapped, 150-199 swapped
            if (GLOBAL_DRIFT_ROUND // self.drift_interval) % 2 == 1:
                partner_sample = self._get_from_partner(idx)
                if partner_sample is not None:
                    x, y = partner_sample[0], partner_sample[1]

        # Mild drift: apply to image tensors
        if self.drift_type in ('slight', 'both'):
            try:
                if torch.is_tensor(x) and x.ndim >= 2:
                    x = self._apply_mild(x)
            except Exception:
                pass

        return x, y


def _apply_mild_tensor(x, drift_step, noise_step, noise_max, rotation_step):
    if drift_step <= 0:
        return x
    out = x
    if noise_step > 0.0:
        noise_std = min(noise_max, drift_step * noise_step)
        if noise_std > 0.0:
            out = out + noise_std * torch.randn_like(out)
    if abs(rotation_step) > 0.0 and torch.is_tensor(out) and out.ndim >= 2:
        angle = drift_step * rotation_step
        if abs(angle) > 0.0:
            try:
                from torchvision.transforms.functional import rotate
                out = rotate(out, angle=angle)
            except Exception:
                pass
    return out


class TemporalDriftTensorClientData:
    """A light-weight proxy for tensor-based clients (e.g. IFCA).

    It exposes the same attributes as the original client data object but
    dynamically swaps whole client tensors when heavy drift is active.
    """

    def __init__(self, train_x, train_y, test_x, test_y, client_id=0, cluster_id=-1,
                 drift_type='none', drift_every=5, noise_step=0.01, noise_max=0.10,
                 rotation_step=5.0, drift_interval=25, partner_map=None, all_client_data=None):
        self._base_train_x = train_x
        self._base_train_y = train_y
        self._base_test_x = test_x
        self._base_test_y = test_y
        self.client_id = int(client_id)
        self.cluster_id = int(cluster_id)
        self.drift_type = str(drift_type)
        self.drift_every = max(1, int(drift_every))
        self.noise_step = float(noise_step)
        self.noise_max = float(noise_max)
        self.rotation_step = float(rotation_step)
        self.drift_interval = int(drift_interval)
        self.partner_map = dict(partner_map) if partner_map else {}
        self.all_client_data = all_client_data

    def _partner_payload(self, train=True):
        if not self.partner_map or not self.all_client_data:
            return None
        if self.client_id not in self.partner_map:
            return None
        partner_id = int(self.partner_map[self.client_id])
        if partner_id < 0 or partner_id >= len(self.all_client_data):
            return None
        partner = self.all_client_data[partner_id]
        if train:
            return partner[0], partner[1]
        return partner[2], partner[3]

    def _current_payload(self, train=True):
        if train:
            x, y = self._base_train_x, self._base_train_y
        else:
            x, y = self._base_test_x, self._base_test_y

        if self.drift_type in ('heavy', 'both') and self.drift_interval > 0:
            if (GLOBAL_DRIFT_ROUND // self.drift_interval) % 2 == 1:
                partner_payload = self._partner_payload(train=train)
                if partner_payload is not None:
                    x, y = partner_payload

        if self.drift_type in ('slight', 'both') and torch.is_tensor(x):
            drift_step = GLOBAL_DRIFT_ROUND // self.drift_every
            x = _apply_mild_tensor(x, drift_step, self.noise_step, self.noise_max, self.rotation_step)

        return x, y

    @property
    def train_x(self):
        return self._current_payload(train=True)[0]

    @property
    def train_y(self):
        return self._current_payload(train=True)[1]

    @property
    def test_x(self):
        return self._current_payload(train=False)[0]

    @property
    def test_y(self):
        return self._current_payload(train=False)[1]

    @property
    def train_samples(self):
        return int(self.train_x.shape[0])

    @property
    def test_samples(self):
        return int(self.test_x.shape[0])


def build_temporal_drift_tensor_clients(raw_clients, drift_type='none', drift_every=5,
                                        noise_step=0.01, noise_max=0.10, rotation_step=5.0,
                                        drift_interval=25, swap_spec='', cluster_attr='cluster_id'):
    """Wrap a list of tensor-based client data objects with the same drift policy."""
    base_pool = [
        (client.train_x, client.train_y, client.test_x, client.test_y)
        for client in raw_clients
    ]
    partner_map = build_partner_map_from_swap_spec(swap_spec)
    wrapped_clients = []
    for cid, client in enumerate(raw_clients):
        wrapped_clients.append(
            TemporalDriftTensorClientData(
                train_x=client.train_x,
                train_y=client.train_y,
                test_x=client.test_x,
                test_y=client.test_y,
                client_id=cid,
                cluster_id=int(getattr(client, cluster_attr, -1)),
                drift_type=drift_type,
                drift_every=drift_every,
                noise_step=noise_step,
                noise_max=noise_max,
                rotation_step=rotation_step,
                drift_interval=drift_interval,
                partner_map=partner_map,
                all_client_data=base_pool,
            )
        )
    return wrapped_clients, partner_map


