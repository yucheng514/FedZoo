import time
from collections import defaultdict

import numpy as np
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
            inner_lr=args.local_learning_rate,
            first_order=args.mcfl_first_order,
            local_epochs=args.local_epochs,
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


def run_cfl(args):
    from clients.clientCFL import CFLClient
    from dataset.cfl_emnist import make_cfl_partition
    from models.cfl_models import CFLConvNet
    from servers.serverCFL import CFLServer
    from utils.cfl_helper import ExperimentLogger, display_train_stats

    set_seed(args.cfl_seed)

    client_data, test_data, _ = make_cfl_partition(args)

    model_fn = lambda: CFLConvNet(num_classes=args.num_classes)
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=args.local_learning_rate, momentum=args.cfl_momentum)

    clients = [
        CFLClient(
            args=args,
            idnum=i,
            data=dat,
            model_fn=model_fn,
            optimizer_fn=optimizer_fn,
            batch_size=args.batch_size,
            train_frac=args.cfl_train_frac,
            seed=args.cfl_seed,
        )
        for i, dat in enumerate(client_data)
    ]
    server = CFLServer(global_model=model_fn(), test_data=test_data, device=args.device)

    cfl_stats = ExperimentLogger()
    cluster_indices = [np.arange(len(clients)).astype(int)]
    acc_clients = [client.evaluate() for client in clients]

    for c_round in range(1, args.global_rounds + 1):
        if c_round == 1:
            server.synchronize_clients(clients)

        participating_clients = server.select_clients(clients, frac=1.0)
        for client in participating_clients:
            client.compute_weight_update(epochs=args.local_epochs)
            client.reset()

        similarities = server.compute_pairwise_similarities(clients)
        cluster_indices_new = []
        last_mean_norm = 0.0
        last_max_norm = 0.0

        for idc in cluster_indices:
            current_cluster = [clients[i] for i in idc]
            last_max_norm = server.compute_max_update_norm(current_cluster)
            last_mean_norm = server.compute_mean_update_norm(current_cluster)

            if last_mean_norm < args.cfl_eps_1 and last_max_norm > args.cfl_eps_2 and len(idc) > 2 and c_round > args.cfl_split_round:
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.cluster_clients(similarities[idc][:, idc])
                cluster_indices_new += [idc[c1], idc[c2]]
                cfl_stats.log({"split": c_round})
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate() for client in clients]
        cfl_stats.log(
            {
                "acc_clients": acc_clients,
                "mean_norm": last_mean_norm,
                "max_norm": last_max_norm,
                "rounds": c_round,
                "clusters": cluster_indices,
            }
        )

        cluster_hist = {i: len(idcs) for i, idcs in enumerate(cluster_indices)}
        print(
            f"Round {c_round:03d} | "
            f"acc_mean={np.mean(acc_clients):.4f} | "
            f"mean_norm={last_mean_norm:.4f} | "
            f"max_norm={last_max_norm:.4f} | "
            f"clusters={cluster_hist}"
        )

        if args.cfl_plot_every and c_round % args.cfl_plot_every == 0:
            display_train_stats(cfl_stats, args.cfl_eps_1, args.cfl_eps_2, args.global_rounds)

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    return server, clients, cfl_stats


def run(args):
    if args.algorithm == "CFL":
        run_cfl(args)
    elif args.algorithm == "MCFL":
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