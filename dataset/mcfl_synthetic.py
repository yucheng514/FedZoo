import torch
from torch.utils.data import DataLoader, TensorDataset

from clients.clientMCFL import MCFLClient


# Build non-IID synthetic support/query datasets for each MCFL client.
def make_synthetic_clients(args):
    clients = []

    for cid in range(args.num_clients):
        group = cid % args.mcfl_true_groups
        shift = group * 1.2

        x_support = torch.randn(args.mcfl_samples_per_client, args.mcfl_input_dim) + shift
        x_query = torch.randn(args.mcfl_samples_per_client, args.mcfl_input_dim) + shift

        weight = torch.zeros(args.mcfl_input_dim)
        weight[group::args.mcfl_true_groups] = 1.0

        logits_s = x_support @ weight + 0.2 * torch.randn(args.mcfl_samples_per_client)
        logits_q = x_query @ weight + 0.2 * torch.randn(args.mcfl_samples_per_client)

        y_support = (logits_s > logits_s.median()).long()
        y_query = (logits_q > logits_q.median()).long()

        support_loader = DataLoader(
            TensorDataset(x_support, y_support),
            batch_size=args.batch_size,
            shuffle=True,
        )
        query_loader = DataLoader(
            TensorDataset(x_query, y_query),
            batch_size=args.batch_size,
            shuffle=True,
        )

        clients.append(
            MCFLClient(
                client_id=cid,
                support_loader=support_loader,
                query_loader=query_loader,
                device=args.device,
            )
        )

    return clients

