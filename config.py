import argparse
import os

import torch


def build_parser():
    parser = argparse.ArgumentParser()
    # Shared arguments used by all algorithms.
    parser.add_argument('-al', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-dev', "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-nc', "--num_clients", type=int, default=20)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-tr', "--train_frac", type=float, default=0.8, help="Train fraction used by CFL client split")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('--log_file', type=str, default='', help='Optional log file path for cross-platform stdout tee')
    parser.add_argument('--log_append', action='store_true', help='Append to log file instead of overwrite')

    # FedAvg-specific arguments.
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="For auto_break")
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)

    # MCFL-specific arguments.
    parser.add_argument('--mcfl_seed', type=int, default=42)
    parser.add_argument('--mcfl_backbone', type=str, default='auto', choices=['auto', 'mlp', 'cnn'])
    parser.add_argument('--mcfl_input_dim', type=int, default=64, help='Used by the synthetic fallback benchmark.')
    parser.add_argument('--mcfl_samples_per_client', type=int, default=512, help='Used by the synthetic fallback benchmark.')
    parser.add_argument('--mcfl_true_groups', type=int, default=4, help='Used by the synthetic fallback benchmark.')
    parser.add_argument('--mcfl_hidden_dim', type=int, default=256, help='Used by the synthetic fallback benchmark.')
    parser.add_argument('--mcfl_num_clusters', type=int, default=4)
    parser.add_argument('--mcfl_encoder_embed_dim', type=int, default=64)
    parser.add_argument('--mcfl_outer_lr', type=float, default=1e-3)
    parser.add_argument('--mcfl_support_ratio', type=float, default=0.8)
    parser.add_argument('--mcfl_first_order', type=bool, default=True)
    parser.add_argument('--mcfl_recluster_every', type=int, default=5)

    # CFL-specific arguments.
    parser.add_argument('--cfl_seed', type=int, default=42)
    parser.add_argument('--cfl_data_root', type=str, default='.')
    parser.add_argument('--cfl_split', type=str, default='byclass')
    parser.add_argument('--cfl_dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--cfl_train_samples', type=int, default=10000)
    parser.add_argument('--cfl_test_samples', type=int, default=10000)
    parser.add_argument('--cfl_rotation_clients', type=int, default=5)
    parser.add_argument('--cfl_rotation_degrees', type=int, default=180)
    parser.add_argument('--cfl_momentum', type=float, default=0.9)
    parser.add_argument('--cfl_eps_1', type=float, default=0.4)
    parser.add_argument('--cfl_eps_2', type=float, default=1.6)
    parser.add_argument('--cfl_split_round', type=int, default=20)
    parser.add_argument('--cfl_plot_every', type=int, default=0, help='Plot CFL stats every N rounds; 0 disables plotting.')
    parser.add_argument('--cfl_download', dest='cfl_download', action='store_true', default=True)
    parser.add_argument('--cfl_no_download', dest='cfl_download', action='store_false')
    return parser


def get_args():
    args = build_parser().parse_args()
    validate_args(args)
    return args


def validate_args(args):
    required_common = [
        "algorithm", "device", "dataset", "num_classes", "global_rounds",
        "num_clients", "batch_size", "local_epochs", "local_learning_rate",
    ]
    required_by_algo = {
        "FedAvg": ["model", "join_ratio", "eval_gap"],
        "MCFL": ["mcfl_seed", "mcfl_backbone", "mcfl_hidden_dim", "mcfl_num_clusters"],
        "CFL": ["cfl_seed", "cfl_split", "cfl_dirichlet_alpha", "cfl_momentum", "train_frac"],
    }

    missing = [name for name in required_common if not hasattr(args, name)]
    missing += [name for name in required_by_algo.get(args.algorithm, []) if not hasattr(args, name)]

    if missing:
        raise ValueError(
            f"Missing CLI args for algorithm={args.algorithm}: {sorted(set(missing))}. "
            "Check config.build_parser() argument definitions."
        )


def resolve_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif mps_available:
            args.device = "mps"
        else:
            args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if mps_available else "cpu"
    elif args.device == "mps" and not mps_available:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args
