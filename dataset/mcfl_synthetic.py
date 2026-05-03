from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from clients.clientMCFL import MCFLClient
from dataset.shared_fl import has_partitioned_data
from utils.data_utils import canonical_dataset_name
from utils.data_utils import read_client_data


IMAGE_DATASETS = {"MNIST", "CIFAR10", "Cifar10", "EMNIST"}


def _stack_image_samples(samples):
    xs, ys = [], []
    for x, y in samples:
        if isinstance(x, (tuple, list)):
            return None
        xs.append(x.float())
        ys.append(int(y.item()) if torch.is_tensor(y) else int(y))

    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def _stack_samples_for_backbone(samples, use_cnn, label_to_index=None):
    xs, ys = [], []
    for x, y in samples:
        if isinstance(x, (tuple, list)):
            return None, None

        x = x.float()
        if use_cnn:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim == 1:
                raise ValueError("CNN backbone requires image-like tensors, but got 1D features.")
        else:
            x = x.reshape(-1)

        raw_label = int(y.item()) if torch.is_tensor(y) else int(y)
        if raw_label < 0:
            raise ValueError(f"Negative labels are not supported for MCFL: {raw_label}")

        if label_to_index is not None:
            if raw_label not in label_to_index:
                label_to_index[raw_label] = len(label_to_index)
            label = label_to_index[raw_label]
        else:
            label = raw_label

        xs.append(x)
        ys.append(label)

    x_tensor = torch.stack(xs, dim=0)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor), int(x_tensor[0].numel()) if not use_cnn else int(x_tensor.shape[1] * x_tensor.shape[2] * x_tensor.shape[3])


def _split_dataset(dataset, support_ratio, seed, min_query_size=1):
    n_samples = len(dataset)
    if n_samples < 2:
        raise ValueError("Need at least two samples to split support/query sets.")

    support_size = int(round(n_samples * support_ratio))
    support_size = max(1, min(support_size, n_samples - min_query_size))
    query_size = n_samples - support_size
    if query_size < min_query_size:
        query_size = min_query_size
        support_size = n_samples - query_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)
    support_idx = indices[:support_size]
    query_idx = indices[support_size:support_size + query_size]

    support_subset = torch.utils.data.Subset(dataset, support_idx.tolist())
    query_subset = torch.utils.data.Subset(dataset, query_idx.tolist())
    return support_subset, query_subset


def _build_loaders_from_dataset(dataset, batch_size, support_ratio, seed):
    support_dataset, query_dataset = _split_dataset(dataset, support_ratio, seed)
    support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return support_loader, query_loader


def _make_real_clients(args):
    clients = []
    dataset_name = canonical_dataset_name(args.dataset)
    use_cnn = args.mcfl_backbone == "cnn" or (args.mcfl_backbone == "auto" and dataset_name.upper() in IMAGE_DATASETS)
    inferred_input_dim = None
    label_to_index = {}

    for cid in range(args.num_clients):
        train_samples = read_client_data(dataset_name, cid, is_train=True, few_shot=args.few_shot)
        dataset, feature_dim = _stack_samples_for_backbone(
            train_samples,
            use_cnn=use_cnn,
            label_to_index=label_to_index,
        )
        if dataset is None:
            raise ValueError(f"Dataset {dataset_name} is not image-like for MCFL.")

        if not use_cnn and inferred_input_dim is None:
            inferred_input_dim = feature_dim

        support_loader, query_loader = _build_loaders_from_dataset(
            dataset=dataset,
            batch_size=args.batch_size,
            support_ratio=args.mcfl_support_ratio,
            seed=args.mcfl_seed + cid,
        )
        test_samples = read_client_data(dataset_name, cid, is_train=False, few_shot=args.few_shot)
        test_dataset, _ = _stack_samples_for_backbone(
            test_samples,
            use_cnn=use_cnn,
            label_to_index=label_to_index,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        clients.append(
            MCFLClient(
                client_id=cid,
                support_loader=support_loader,
                query_loader=query_loader,
                test_loader=test_loader,
                device=getattr(args, "mcfl_client_device_resolved", args.device),
                local_epochs=args.local_epochs,
            )
        )

    if inferred_input_dim is not None:
        args.mcfl_input_dim = inferred_input_dim

    inferred_num_classes = max(len(label_to_index), 1)
    if args.num_classes != inferred_num_classes:
        print(
            f"[MCFL] Adjust num_classes from {args.num_classes} to {inferred_num_classes} "
            f"based on observed client labels: {sorted(label_to_index.keys())}"
        )
        args.num_classes = inferred_num_classes

    return clients


def _make_synthetic_clients(args):
    clients = []
    total_samples = max(args.mcfl_samples_per_client, args.batch_size * 8)

    for cid in range(args.num_clients):
        group = cid % args.mcfl_true_groups
        group_shift = torch.randn(args.mcfl_input_dim) * 0.25 + group * 0.75

        x = torch.randn(total_samples, args.mcfl_input_dim) + group_shift
        shared_basis = torch.randn(args.mcfl_input_dim, args.num_classes)
        class_bias = torch.zeros(args.num_classes)
        class_bias[group % args.num_classes] = 1.5
        logits = x @ shared_basis + class_bias + 0.15 * torch.randn(total_samples, args.num_classes)
        y = logits.argmax(dim=1)

        dataset = TensorDataset(x, y)
        support_loader, query_loader = _build_loaders_from_dataset(
            dataset=dataset,
            batch_size=args.batch_size,
            support_ratio=args.mcfl_support_ratio,
            seed=args.mcfl_seed + cid,
        )

        clients.append(
            MCFLClient(
                client_id=cid,
                support_loader=support_loader,
                query_loader=query_loader,
                test_loader=query_loader,
                device=getattr(args, "mcfl_client_device_resolved", args.device),
                local_epochs=args.local_epochs,
            )
        )

    return clients


def make_mcfl_clients(args):
    dataset_name = canonical_dataset_name(args.dataset)
    if dataset_name.upper() in IMAGE_DATASETS:
        try:
            if has_partitioned_data(dataset_name) or Path(f"./dataset/data/{dataset_name}").exists():
                return _make_real_clients(args)
        except Exception as exc:
            print(f"[MCFL] Falling back to synthetic benchmark: {exc}")

    return _make_synthetic_clients(args)


# Backward-compatible alias.
def make_synthetic_clients(args):
    return make_mcfl_clients(args)
