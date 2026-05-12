import time
from collections import defaultdict
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import torch
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from config import get_args, resolve_device
from dataset.mcfl_synthetic import make_mcfl_clients
from dataset.shared_fl import has_partitioned_data, make_partitioned_tensor_clients
from models.mcfl_models import MCFLMLPClassifier
from servers.serverMCFL import MCFLServer
from utils.mcfl_utils import set_seed
from string import ascii_lowercase
from utils.data_utils import set_global_drift_round

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


def print_python_command():
    """输出完整的python命令行"""
    command = "python " + " ".join(sys.argv)
    print("=" * 80)
    print("COMMAND LINE:")
    print(command)
    print("=" * 80)


def format_timezone_now(offset_hours=8):
    tz = timezone(timedelta(hours=offset_hours))
    now = datetime.now(tz)
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC+08:00"


def print_experiment_start_time():
    print(f"Experiment start time (UTC+8): {format_timezone_now(8)}")


def print_experiment_end_time():
    print(f"Experiment end time (UTC+8): {format_timezone_now(8)}")


def print_run_summary(args):
    mcfl_client_device = getattr(args, "mcfl_client_device_resolved", None)
    summary = (
        f"Run: algorithm={args.algorithm} | dataset={args.dataset} | device={args.device} | "
        f"rounds={args.global_rounds} | clients={args.num_clients} | local_epochs={args.local_epochs} | "
        f"lr={args.local_learning_rate} | batch_size={args.batch_size}"
    )
    if args.algorithm == "MCFL" and mcfl_client_device is not None:
        summary += f" | mcfl_client_device={mcfl_client_device}"
    print(summary)

    if getattr(args, "print_args", False):
        print("=== args " + "=" * 50)
        for arg in vars(args):
            print(arg, '=', getattr(args, arg))
        print("=== args end " + "=" * 46)


def run_fedavg(args):
    from models.models import FedAvgCNN
    from servers.serverAvg import FedAvg

    set_seed(args.seed)
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

    # 默认将客户端设备和主设备保持一致（在有 GPU 的情况下让客户端使用 GPU），
    # 如果用户明确传入 --mcfl_client_device 可覆盖该行为。
    if args.mcfl_client_device == "auto":
        # 如果主设备是 cuda 或 mps，则默认让客户端也使用相同设备以提升并行计算效率。
        args.mcfl_client_device_resolved = args.device
    else:
        args.mcfl_client_device_resolved = args.mcfl_client_device

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
        selected_backbone = "cnn" if input_is_image else "mlp"
    else:
        selected_backbone = args.mcfl_backbone

    if selected_backbone == "cnn":
        if not input_is_image:
            raise ValueError(
                f"MCFL backbone is set to {selected_backbone}, but client data is not image-like "
                f"(got shape {tuple(warmup_x.shape)}). "
                "Use --mcfl_backbone mlp or provide image-format client data."
            )
        use_image_backbone = True
    else:
        use_image_backbone = False

    if use_image_backbone:
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

    # 处理初始簇数参数（向后兼容）
    initial_clusters = args.mcfl_initial_clusters if args.mcfl_initial_clusters is not None else args.mcfl_num_clusters
    
    server = MCFLServer(
        global_model=base_model,
        num_clusters=initial_clusters,
        encoder_embed_dim=args.mcfl_encoder_embed_dim,
        outer_lr=args.mcfl_outer_lr,
        model_mix=args.mcfl_model_mix,
        device=args.device,
        total_rounds=args.global_rounds,
        recluster_every=args.mcfl_recluster_every,
        recluster_warmup_rounds=args.mcfl_recluster_warmup_rounds,
        stop_recluster_after=args.mcfl_stop_recluster_after,
        max_reclusters=args.mcfl_max_reclusters,
        skip_final_recluster=args.mcfl_skip_final_recluster,
        cluster_change_threshold=args.mcfl_cluster_change_threshold,
        cluster_method=args.mcfl_cluster_method,
        cluster_feature=args.mcfl_cluster_feature,
        algorithm=args.mcfl_algorithm,
        enable_dynamic_clustering=args.mcfl_enable_dynamic_clustering,
        outlier_threshold=args.mcfl_outlier_threshold,
        drift_severity_low=args.mcfl_drift_severity_low,
        drift_severity_high=args.mcfl_drift_severity_high,
        outlier_pooling_threshold=args.mcfl_outlier_pooling_threshold,
        agglomerative_threshold=args.mcfl_agglomerative_threshold,
        global_reg=args.mcfl_global_reg,
    )

    server.assign_initial_clusters(clients)
    best_test_acc = float("-inf")

    for rnd in range(args.global_rounds):
        print(f"==================== Round {rnd:03d} start ====================")
        # Inform DriftDataset and clients of the current global round
        set_global_drift_round(rnd)
        
        drift_interval = getattr(args, 'drift_interval', 25)
        if getattr(args, 'drift_type', 'none') in ('heavy', 'both') and drift_interval > 0:
            if rnd > 0 and rnd % drift_interval == 0:
                print(f"Round {rnd}: Triggering Heavy Concept Drift!")
                
        for c in clients:
            try:
                c.current_round = rnd
            except Exception:
                pass
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

        cluster_clients = defaultdict(list)
        for client in clients:
            cluster_clients[client.cluster_id].append(client.client_id)

        # map numeric cluster ids to letters absolutely to match log printing logic
        unique_ids = sorted(cluster_clients.keys())
        id_to_letter = {cid: ascii_lowercase[cid] if cid < len(ascii_lowercase) else str(cid) for cid in unique_ids}
        
        cluster_hist_pretty = {}
        for cid, client_list in cluster_clients.items():
            letter = id_to_letter.get(cid, str(cid))
            cluster_hist_pretty[letter] = f"{sorted(client_list)} (count: {len(client_list)})"

        # 针对字母顺序进行排序，使得打印更加清晰
        cluster_hist_pretty = dict(sorted(cluster_hist_pretty.items()))

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

        dynamic_suffix = ""
        if getattr(args, "mcfl_enable_dynamic_clustering", False):
            dynamic_summary = getattr(
                server,
                "last_dynamic_cluster_summary",
                {"outliers": 0, "new_clusters": 0, "total_clusters": len(server.cluster_models), "reassigned": 0},
            )
            merged_count = dynamic_summary.get('merged_singletons', 0)
            dynamic_suffix = (
                f"outliers={dynamic_summary['outliers']} | "
                f"new_clusters={dynamic_summary['new_clusters']} | "
                f"merged_singletons={merged_count} | "
                f"total_clusters={dynamic_summary['total_clusters']} | "
                f"reassigned={dynamic_summary['reassigned']} | "
            )

        mean_adapted_test_acc = sum(adapted_test_accs) / len(adapted_test_accs)
        mean_raw_test_acc = sum(raw_test_accs) / len(raw_test_accs)

        if getattr(args, 'wandb', False) and wandb is not None:
            wandb.log({
                "round": rnd,
                "test_acc": mean_adapted_test_acc,
                "raw_test_acc": mean_raw_test_acc,
                "support_loss": avg_support,
                "query_loss": avg_query,
                "support_acc": support_acc,
                "query_acc": query_acc,
            })

        print(
            f"Round {rnd:03d} | "
            f"support_loss={avg_support:.4f} | "
            f"query_loss={avg_query:.4f} | "
            f"support_acc={support_acc:.4f} | "
            f"query_acc={query_acc:.4f} | "
            f"test_acc={mean_adapted_test_acc:.4f} | "
            f"raw_test_acc={mean_raw_test_acc:.4f} | "
            f"{dynamic_suffix}"
            f"clusters={cluster_hist_pretty}"
        )
        best_test_acc = max(best_test_acc, mean_adapted_test_acc)
        budget.append(time.time() - s_t)
        print(f"time cost:{budget[-1]:.2f}")
        print(f"==================== Round {rnd:03d} end ====================")

    if budget:
        print("\nBest accuracy.")
        print(best_test_acc if best_test_acc != float("-inf") else 0.0)
        print("\nAverage time cost per round.")
        print(f"{sum(budget) / len(budget):.2f}")


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
        set_global_drift_round(c_round)
        
        drift_interval = getattr(args, 'drift_interval', 25)
        if getattr(args, 'drift_type', 'none') in ('heavy', 'both') and drift_interval > 0:
            if c_round > 0 and c_round % drift_interval == 0:
                print(f"Round {c_round}: Triggering Heavy Concept Drift!")
                
        print(f"==================== Round {c_round:03d} start ====================")
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

        mean_acc = np.mean(acc_clients)
        if getattr(args, 'wandb', False):
            wandb.log({
                "round": c_round,
                "test_acc": mean_acc,
                "mean_norm": last_mean_norm,
                "max_norm": last_max_norm,
                "num_clusters": len(cluster_indices),
            })

        cluster_hist = {i: len(idcs) for i, idcs in enumerate(cluster_indices)}
        print(
            f"Round {c_round:03d} | "
            f"acc_mean={mean_acc:.4f} | "
            f"mean_norm={last_mean_norm:.4f} | "
            f"max_norm={last_max_norm:.4f} | "
            f"clusters={cluster_hist}"
        )

        print(f"==================== Round {c_round:03d} end ====================")

        if args.cfl_plot_every and c_round % args.cfl_plot_every == 0:
            display_train_stats(cfl_stats, args.cfl_eps_1, args.cfl_eps_2, args.global_rounds)

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    return server, clients, cfl_stats


def run_ifca(args):
    from clients.clientIFCA import IFCAClient
    from models.ifca_models import IFCALinearRegressor, IFCAMLPClassifier, IFCASmallCNN
    from servers.serverIFCA import IFCAServer
    from utils.data_utils import build_temporal_drift_tensor_clients, set_global_drift_round

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

    if getattr(args, 'drift_type', 'none') != 'none':
        raw_clients, _ = build_temporal_drift_tensor_clients(
            raw_clients,
            drift_type=getattr(args, 'drift_type', 'none'),
            drift_every=getattr(args, 'drift_every', 5),
            noise_step=getattr(args, 'drift_noise_step', 0.01),
            noise_max=getattr(args, 'drift_noise_max', 0.10),
            rotation_step=getattr(args, 'drift_rotation_step', 5.0),
            drift_interval=getattr(args, 'drift_interval', 25),
            swap_spec=getattr(args, 'drift_swap_clients', ''),
        )

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
    set_global_drift_round(0)
    initial = server.evaluate()
    initial_cluster = "n/a" if initial["cluster_acc"] < 0 else f"{initial['cluster_acc']:.4f}"
    print(f"Round -01 | train_loss={initial['train_loss']:.4f} | cluster_acc={initial_cluster} | assignments={initial['assignment_hist']}")
    if task == "regression":
        print(f"Round -01 | test_mse={initial['test_mse']:.4f}")
    else:
        print(f"Round -01 | test_acc={initial['test_acc']:.4f}")
    initial_time = time.time() - initial_s_t
    print(f"time cost:{initial_time:.2f}")
    budget.append(initial_time)
    best_test_acc = initial["test_acc"] if task != "regression" else float("-inf")

    if getattr(args, 'wandb', False):
        if task == "regression":
            wandb.log({"round": -1, "test_mse": initial['test_mse'], "train_loss": initial['train_loss']})
        else:
            wandb.log({"round": -1, "test_acc": initial['test_acc'], "train_loss": initial['train_loss']})

    for rnd in range(args.global_rounds):
        set_global_drift_round(rnd)
        print(f"==================== Round {rnd:03d} start ====================")
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

        if getattr(args, 'wandb', False):
            if task == "regression":
                wandb.log({"round": rnd, "test_mse": eval_stats['test_mse'], "train_loss": eval_stats['train_loss']})
            else:
                wandb.log({"round": rnd, "test_acc": eval_stats['test_acc'], "train_loss": eval_stats['train_loss']})

        if task != "regression":
            best_test_acc = max(best_test_acc, eval_stats["test_acc"])
        budget.append(time.time() - s_t)
        print(f"time cost:{budget[-1]:.2f}")
        print(f"==================== Round {rnd:03d} end ====================")

    if budget:
        print("\nBest accuracy.")
        if task == "regression":
            print("n/a")
        else:
            print(best_test_acc)
        print("\nAverage time cost per round.")
        print(f"{sum(budget) / len(budget):.2f}")

    if args.ifca_checkpoint:
        checkpoint_path = Path(args.ifca_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cluster_models": [model.cpu().state_dict() for model in server.cluster_models]}, checkpoint_path)

    return server


def run_perfedavg(args):
    """运行 Per-FedAvg 算法"""
    from models.models import FedAvgCNN
    from servers.serverPerFedAvg import serverPerFedAvg

    set_seed(args.seed)
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

    set_seed(args.seed)
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


def run_server_base(server_cls, args):
    """Fallback runner for server_cls that inherits from ServerBase (like FedAvg)"""
    from models.models import FedAvgCNN

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

        server = server_cls(args, i)
        server.train()
        time_list.append(time.time() - start)


def run(args):
    if getattr(args, 'wandb', False) and wandb is not None:
        wandb.init(
            project="FedZoo",
            name=f"{args.algorithm}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

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
    else: #fedavg
        run_fedavg(args)
        
    if getattr(args, 'wandb', False) and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    total_start = time.time()
    args = get_args()
    resolve_device(args)
    
    if not args.log_file:
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = f"logs/{args.algorithm}_{now_str}.log"

    with maybe_log_to_file(args.log_file, append=args.log_append):
        print_experiment_start_time()
        print_python_command()
        print_run_summary(args)
        try:
            run(args)
        finally:
            print_experiment_end_time()
            total_runtime = time.time() - total_start
            print(f"Total runtime: {total_runtime:.2f} seconds")
