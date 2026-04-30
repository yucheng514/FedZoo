import time
from collections import defaultdict
from contextlib import contextmanager
import copy
from pathlib import Path
import sys

import numpy as np
import torch

from config import get_args, resolve_device
from dataset.mcfl_synthetic import make_mcfl_clients
from dataset.shared_fl import has_partitioned_data, make_partitioned_tensor_clients
from models.mcfl_models import MCFLMLPClassifier
from servers.serverMCFL import MCFLServer
from utils.mcfl_utils import set_seed

torch.manual_seed(0)


def select_fractional_clients(clients, frac, seed):
    if not clients:
        return []
    frac = min(max(frac, 0.0), 1.0)
    n_selected = max(1, int(round(len(clients) * frac)))
    n_selected = min(len(clients), n_selected)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(clients), size=n_selected, replace=False)
    return [clients[i] for i in indices]


class StdoutTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def maybe_log_to_file(log_file, append=False):
    if not log_file:
        yield
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fp:
        original_stdout = sys.stdout
        sys.stdout = StdoutTee(original_stdout, fp)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def run_fedavg(args):
    from models.models import FedAvgCNN
    from servers.serverAvg import FedAvg

    time_list = []
    model_str = args.model
    dataset_name = args.dataset.upper()

    for i in range(0, 1):
        print("Creating server and clients ...")
        start = time.time()
        if model_str == "CNN":
            if dataset_name in {"MNIST", "EMNIST", "FEMNIST"}:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif dataset_name == "CIFAR10":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        server = FedAvg(args, i)
        server.train()
        time_list.append(time.time() - start)


def run_mcfl(args):
    from models.models import FedAvgCNN

    set_seed(args.mcfl_seed)

    clients = make_mcfl_clients(args)
    budget = []

    warmup_x, _ = next(iter(clients[0].support_loader))

    def _infer_fedavgcnn_dim(sample_batch):
        h, w = int(sample_batch.shape[-2]), int(sample_batch.shape[-1])
        h = (h - 4) // 2
        h = (h - 4) // 2
        w = (w - 4) // 2
        w = (w - 4) // 2
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid input shape for FedAvgCNN: {tuple(sample_batch.shape)}")
        return 64 * h * w

    input_is_image = warmup_x.ndim == 4
    if args.mcfl_backbone == "auto":
        use_cnn = input_is_image
    elif args.mcfl_backbone == "cnn":
        if not input_is_image:
            raise ValueError(
                "MCFL backbone is set to cnn, but client data is not image-like "
                f"(got shape {tuple(warmup_x.shape)}). "
                "Use --mcfl_backbone mlp or provide image-format client data."
            )
        use_cnn = True
    else:
        use_cnn = False

    if use_cnn:
        in_features = int(warmup_x.shape[1])
        conv_dim = _infer_fedavgcnn_dim(warmup_x)
        base_model = FedAvgCNN(in_features=in_features, num_classes=args.num_classes, dim=conv_dim)
    else:
        base_model = MCFLMLPClassifier(
            in_dim=None,
            hidden_dim=args.mcfl_hidden_dim,
            num_classes=args.num_classes,
        )

    base_model = base_model.to(args.device)

    # Materialize lazy layers / verify the backbone shape with one real client batch.
    with torch.no_grad():
        base_model(warmup_x.to(args.device))

    server = MCFLServer(
        global_model=base_model,
        num_clusters=args.mcfl_num_clusters,
        encoder_embed_dim=args.mcfl_encoder_embed_dim,
        outer_lr=args.mcfl_outer_lr,
        device=args.device,
        recluster_every=args.mcfl_recluster_every,
        cluster_feature=args.mcfl_cluster_feature,
    )

    server.assign_initial_clusters(clients)
    best_test_acc = float("-inf")

    for rnd in range(args.global_rounds):
        s_t = time.time()
        participating_clients = select_fractional_clients(clients, args.join_ratio, args.mcfl_seed + rnd)
        stats = server.train_round(
            participating_clients,
            round_idx=rnd,
            inner_lr=args.local_learning_rate,
            first_order=args.mcfl_first_order,
            local_epochs=args.local_epochs,
        )

        avg_support = sum(s["support_loss"] for s in stats) / len(stats)
        avg_query = sum(s["query_loss"] for s in stats) / len(stats)
        total_support_samples = sum(s["support_samples"] for s in stats)
        total_query_samples = sum(s["query_samples"] for s in stats)
        support_acc = sum(s["support_correct"] for s in stats) / max(total_support_samples, 1)
        query_acc = sum(s["query_correct"] for s in stats) / max(total_query_samples, 1)

        cluster_hist = defaultdict(int)
        for client in clients:
            cluster_hist[client.cluster_id] += 1

        raw_test_accs = [client.evaluate(server.cluster_models[client.cluster_id], adapt=False) for client in clients]
        adapted_test_accs = [
            client.evaluate(
                server.cluster_models[client.cluster_id],
                adapt=True,
                inner_lr=args.local_learning_rate,
                local_epochs=args.local_epochs,
            )
            for client in clients
        ]

        print(
            f"Round {rnd:03d} | "
            f"support_loss={avg_support:.4f} | "
            f"query_loss={avg_query:.4f} | "
            f"support_acc={support_acc:.4f} | "
            f"query_acc={query_acc:.4f} | "
            f"test_acc={sum(adapted_test_accs) / len(adapted_test_accs):.4f} | "
            f"raw_test_acc={sum(raw_test_accs) / len(raw_test_accs):.4f} | "
            f"clusters={dict(cluster_hist)}"
        )
        best_test_acc = max(best_test_acc, sum(adapted_test_accs) / len(adapted_test_accs))
        budget.append(time.time() - s_t)
        print('-' * 25, 'time cost', '-' * 25, budget[-1])

    if budget:
        print("\nBest accuracy.")
        print(best_test_acc if best_test_acc != float("-inf") else 0.0)
        print("\nAverage time cost per round.")
        print(sum(budget) / len(budget))


def run_cfl(args):
    from clients.clientCFL import CFLClient
    from dataset.cfl_emnist import make_cfl_partition
    from servers.serverCFL import CFLServer
    from utils.cfl_helper import ExperimentLogger, display_train_stats

    set_seed(args.cfl_seed)

    client_data, test_data, _ = make_cfl_partition(args)

    dataset_name = args.dataset.upper()
    if has_partitioned_data(args.dataset) and dataset_name in {"MNIST", "EMNIST", "FEMNIST", "CIFAR10"}:
        from models.models import FedAvgCNN

        if dataset_name == "CIFAR10":
            model_fn = lambda: FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
        else:
            model_fn = lambda: FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
    else:
        from models.cfl_models import CFLConvNet

        model_fn = lambda: CFLConvNet(
            num_classes=args.num_classes,
            in_channels=3 if dataset_name == "CIFAR10" else 1,
        )
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=args.local_learning_rate, momentum=args.cfl_momentum)

    clients = [
        CFLClient(
            args=args,
            idnum=i,
            data=dat,
            model_fn=model_fn,
            optimizer_fn=optimizer_fn,
            batch_size=args.batch_size,
            train_frac=args.train_frac,
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

        active_clients = server.select_clients(clients, frac=args.join_ratio)
        active_ids = {client.id for client in active_clients}
        for client in clients:
            for tensor in client.dW.values():
                tensor.zero_()
        participating_clients = active_clients
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
        server.aggregate_clusterwise(client_clusters, active_ids=active_ids)

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


def run_ifca(args):
    from clients.clientIFCA import IFCAClient
    from models.ifca_models import IFCALinearRegressor, IFCAMLPClassifier, IFCASmallCNN
    from servers.serverIFCA import IFCAServer

    set_seed(args.ifca_seed)

    dataset_name = args.dataset.upper()
    if dataset_name in {"MNIST", "EMNIST"} and has_partitioned_data(args.dataset):
        from models.models import FedAvgCNN

        raw_clients = make_partitioned_tensor_clients(args, flatten=False)
        cluster_models = [FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024) for _ in range(args.ifca_clusters)]
        criterion = torch.nn.CrossEntropyLoss()
        task = "classification"
    elif dataset_name == "MNIST":
        from dataset.ifca_rotated_mnist import make_ifca_rotated_mnist_clients

        raw_clients = make_ifca_rotated_mnist_clients(args)
        cluster_models = [
            IFCAMLPClassifier(
                input_dim=28 * 28,
                hidden_dim=args.ifca_mnist_hidden_dim,
                num_classes=args.num_classes,
            )
            for _ in range(args.ifca_clusters)
        ]
        criterion = torch.nn.CrossEntropyLoss()
        task = "classification"
    elif dataset_name == "CIFAR10":
        # Check if pre-partitioned CIFAR10 data exists (from dataset generation)
        if has_partitioned_data(args.dataset):
            from dataset.ifca_partitioned_cifar import make_ifca_partitioned_cifar_clients
            raw_clients = make_ifca_partitioned_cifar_clients(args)
            print(f"Using pre-partitioned CIFAR10 data for IFCA (skipping download)")
        else:
            from dataset.ifca_rotated_cifar import make_ifca_rotated_cifar_clients
            raw_clients = make_ifca_rotated_cifar_clients(args)
        cluster_models = [IFCASmallCNN(in_channels=3, num_classes=args.num_classes, classifier_dim=1600) for _ in range(args.ifca_clusters)]
        criterion = torch.nn.CrossEntropyLoss()
        task = "classification"
    elif dataset_name in {"FEMNIST", "EMNIST"}:
        from dataset.ifca_emnist import make_ifca_emnist_clients

        raw_clients = make_ifca_emnist_clients(args)
        cluster_models = [IFCASmallCNN(in_channels=1, num_classes=args.num_classes, classifier_dim=1024) for _ in range(args.ifca_clusters)]
        criterion = torch.nn.CrossEntropyLoss()
        task = "classification"
    elif dataset_name in {"IFCA_SYNTHETIC", "SYNTHETIC"}:
        from dataset.ifca_synthetic import make_ifca_synthetic_clients

        raw_clients, true_params = make_ifca_synthetic_clients(args)
        cluster_models = [IFCALinearRegressor(input_dim=args.ifca_synthetic_dim) for _ in range(args.ifca_clusters)]
        criterion = torch.nn.MSELoss()
        task = "regression"
        print(f"True synthetic cluster parameters: {tuple(true_params.shape)}")
    else:
        raise ValueError(
            f"IFCA currently supports dataset=MNIST, CIFAR10, FEMNIST/EMNIST, or IFCA_SYNTHETIC, got {args.dataset}"
        )

    if args.ifca_mode == "local":
        cluster_models = [copy.deepcopy(cluster_models[0]) for _ in range(len(raw_clients))]
    clients = [IFCAClient(client_id=i, data=data, task=task, device=args.device) for i, data in enumerate(raw_clients)]
    server = IFCAServer(
        cluster_models=[model.to(args.device) for model in cluster_models],
        clients=clients,
        criterion=criterion,
        task=task,
        device=args.device,
        mode=args.ifca_mode,
        freeze_backbone=args.ifca_freeze_backbone,
    )
    budget = []

    if args.ifca_checkpoint and Path(args.ifca_checkpoint).exists():
        saved = torch.load(args.ifca_checkpoint, map_location="cpu")
        for model, state in zip(server.cluster_models, saved["cluster_models"]):
            model.load_state_dict(state)

    if args.ifca_mode == "oneshot":
        server.initialize_fixed_assignments(strategy=args.ifca_init_strategy)
    elif args.ifca_init_rounds > 0:
        if args.ifca_init_strategy == "random":
            assignments = [client.id % len(server.cluster_models) for client in clients]
        else:
            assignments = server.assign_clients()
        server.warmstart_clusters(assignments=assignments, lr=args.local_learning_rate, local_epochs=args.ifca_tau, rounds=args.ifca_init_rounds)

    initial_s_t = time.time()
    initial = server.evaluate()
    initial_cluster = "n/a" if initial["cluster_acc"] < 0 else f"{initial['cluster_acc']:.4f}"
    print(f"Round -01 | train_loss={initial['train_loss']:.4f} | cluster_acc={initial_cluster} | assignments={initial['assignment_hist']}")
    if task == "regression":
        print(f"Round -01 | test_mse={initial['test_mse']:.4f}")
    else:
        print(f"Round -01 | test_acc={initial['test_acc']:.4f}")
    initial_time = time.time() - initial_s_t
    print('-' * 25, 'time cost', '-' * 25, initial_time)
    budget.append(initial_time)
    best_test_acc = initial["test_acc"] if task != "regression" else float("-inf")

    for rnd in range(args.global_rounds):
        s_t = time.time()
        participating_clients = select_fractional_clients(clients, args.join_ratio, args.ifca_seed + rnd)
        server.train_round(lr=args.local_learning_rate, local_epochs=args.ifca_tau, clients=participating_clients)
        eval_stats = server.evaluate()
        cluster_acc_str = "n/a" if eval_stats["cluster_acc"] < 0 else f"{eval_stats['cluster_acc']:.4f}"
        line = (
            f"Round {rnd:03d} | "
            f"train_loss={eval_stats['train_loss']:.4f} | "
            f"cluster_acc={cluster_acc_str} | "
            f"assignments={eval_stats['assignment_hist']}"
        )
        if task == "regression":
            line += f" | test_mse={eval_stats['test_mse']:.4f}"
        else:
            line += f" | test_acc={eval_stats['test_acc']:.4f}"
        print(line)
        if task != "regression":
            best_test_acc = max(best_test_acc, eval_stats["test_acc"])
        budget.append(time.time() - s_t)
        print('-' * 25, 'time cost', '-' * 25, budget[-1])

    if budget:
        print("\nBest accuracy.")
        if task == "regression":
            print("n/a")
        else:
            print(best_test_acc)
        print("\nAverage time cost per round.")
        print(sum(budget) / len(budget))

    if args.ifca_checkpoint:
        checkpoint_path = Path(args.ifca_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cluster_models": [model.cpu().state_dict() for model in server.cluster_models]}, checkpoint_path)

    return server


def run_perfedavg(args):
    """运行 Per-FedAvg 算法"""
    from models.models import FedAvgCNN
    from servers.serverPerFedAvg import serverPerFedAvg

    time_list = []
    model_str = args.model
    dataset_name = args.dataset.upper()

    for i in range(0, 1):
        print("Creating server and clients ...")
        start = time.time()
        if model_str == "CNN":
            if dataset_name in {"MNIST", "EMNIST", "FEMNIST"}:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif dataset_name == "CIFAR10":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        server = serverPerFedAvg(args, i)
        server.train()
        time_list.append(time.time() - start)


def run_pfedme(args):
    """运行 pFedMe 算法"""
    from models.models import FedAvgCNN
    from servers.serverpFedMe import serverpFedMe

    time_list = []
    model_str = args.model
    dataset_name = args.dataset.upper()

    for i in range(0, 1):
        print("Creating server and clients ...")
        start = time.time()
        if model_str == "CNN":
            if dataset_name in {"MNIST", "EMNIST", "FEMNIST"}:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif dataset_name == "CIFAR10":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        server = serverpFedMe(args, i)
        server.train()
        time_list.append(time.time() - start)


def run(args):
    if args.algorithm == "CFL":
        run_cfl(args)
    elif args.algorithm == "MCFL":
        run_mcfl(args)
    elif args.algorithm == "IFCA":
        run_ifca(args)
    elif args.algorithm in {"PerFedAvg", "PerAvg"}:
        run_perfedavg(args)
    elif args.algorithm == "pFedMe":
        run_pfedme(args)
    else:
        run_fedavg(args)


if __name__ == "__main__":
    total_start = time.time()

    args = get_args()
    resolve_device(args)
    with maybe_log_to_file(args.log_file, append=args.log_append):
        print(f"\nUsing device: {args.device}\n")
        print("=== args " + "=" * 50)
        for arg in vars(args):
            print(arg, '=', getattr(args, arg))
        print("=== args end " + "=" * 46)
        run(args)
