import time
from collections import defaultdict

import torch

from config import get_args, resolve_device
from dataset.mcfl_synthetic import make_mcfl_clients
from models.mcfl_models import MCFLMLPClassifier
from servers.serverMCFL import MCFLServer
from utils.mcfl_utils import set_seed

torch.manual_seed(0)


def run_fedavg(args):
    from models.models import FedAvgCNN
    from servers.serverAvg import FedAvg

    time_list = []
    model_str = args.model

    for i in range(0, 1):
        print("Creating server and clients ...")
        start = time.time()
        if model_str == "CNN":
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        server = FedAvg(args, i)
        server.train()
        time_list.append(time.time() - start)


def run_mcfl(args):
    from models.models import FedAvgCNN

    set_seed(args.mcfl_seed)

    clients = make_mcfl_clients(args)

    if args.mcfl_backbone == "cnn" or (args.mcfl_backbone == "auto" and args.dataset in {"MNIST", "Cifar10"}):
        if "MNIST" in args.dataset:
            base_model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
        elif "Cifar10" in args.dataset:
            base_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
        else:
            raise ValueError(f"Unsupported image dataset for MCFL CNN backbone: {args.dataset}")
    else:
        base_model = MCFLMLPClassifier(
            in_dim=None,
            hidden_dim=args.mcfl_hidden_dim,
            num_classes=args.num_classes,
        )

    base_model = base_model.to(args.device)

    # Materialize lazy layers / verify the backbone shape with one real client batch.
    warmup_x, _ = next(iter(clients[0].support_loader))
    with torch.no_grad():
        base_model(warmup_x.to(args.device))

    server = MCFLServer(
        global_model=base_model,
        num_clusters=args.mcfl_num_clusters,
        encoder_embed_dim=args.mcfl_encoder_embed_dim,
        outer_lr=args.mcfl_outer_lr,
        device=args.device,
        recluster_every=args.mcfl_recluster_every,
    )

    server.assign_initial_clusters(clients)

    for rnd in range(args.global_rounds):
        stats = server.train_round(
            clients,
            round_idx=rnd,
            inner_lr=args.mcfl_inner_lr,
            first_order=args.mcfl_first_order,
            local_epochs=args.mcfl_local_epochs,
        )

        avg_support = sum(s["support_loss"] for s in stats) / len(stats)
        avg_query = sum(s["query_loss"] for s in stats) / len(stats)

        cluster_hist = defaultdict(int)
        for client in clients:
            cluster_hist[client.cluster_id] += 1

        print(
            f"Round {rnd:03d} | "
            f"support_loss={avg_support:.4f} | "
            f"query_loss={avg_query:.4f} | "
            f"clusters={dict(cluster_hist)}"
        )


def run(args):
    if args.algorithm == "MCFL":
        run_mcfl(args)
    else:
        run_fedavg(args)


if __name__ == "__main__":
    total_start = time.time()

    args = get_args()
    resolve_device(args)
    print(f"\nUsing device: {args.device}\n")
    print("=== args " + "=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=== args end " + "=" * 46)
    run(args)