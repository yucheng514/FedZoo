from dataclasses import dataclass

import torch


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


def _split_client_tensors(x, y, train_frac):
    split_idx = max(1, min(x.shape[0] - 1, int(round(x.shape[0] * train_frac))))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def make_ifca_synthetic_clients(args):
    if args.num_clients % args.ifca_clusters != 0:
        raise ValueError("IFCA synthetic setup requires num_clients divisible by ifca_clusters.")

    generator = torch.Generator().manual_seed(args.ifca_seed)
    params = []
    for _ in range(args.ifca_clusters):
        mask = torch.bernoulli(
            torch.full((args.ifca_synthetic_dim,), 0.5),
            generator=generator,
        )
        params.append(mask * args.ifca_synthetic_separation)

    clients = []
    clients_per_cluster = args.num_clients // args.ifca_clusters
    for client_idx in range(args.num_clients):
        cluster_id = client_idx // clients_per_cluster
        x = torch.randn(
            args.ifca_synthetic_samples,
            args.ifca_synthetic_dim,
            generator=generator,
        )
        noise = torch.randn(args.ifca_synthetic_samples, generator=generator) * args.ifca_synthetic_noise
        y = x @ params[cluster_id] + noise
        train_x, train_y, test_x, test_y = _split_client_tensors(x, y, args.train_frac)
        clients.append(
            IFCATensorClient(
                train_x=train_x.float(),
                train_y=train_y.float(),
                test_x=test_x.float(),
                test_y=test_y.float(),
                cluster_id=cluster_id,
            )
        )

    return clients, torch.stack(params)
