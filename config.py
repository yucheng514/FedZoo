import argparse
import os

import torch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-al', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-dev', "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Total number of clients")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Multiple update steps in one local epoch.")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="For auto_break")
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)

    # MCFL-related arguments
    parser.add_argument('--mcfl_seed', type=int, default=42)
    parser.add_argument('--mcfl_input_dim', type=int, default=20)
    parser.add_argument('--mcfl_samples_per_client', type=int, default=64)
    parser.add_argument('--mcfl_true_groups', type=int, default=3)
    parser.add_argument('--mcfl_hidden_dim', type=int, default=64)
    parser.add_argument('--mcfl_num_clusters', type=int, default=3)
    parser.add_argument('--mcfl_encoder_embed_dim', type=int, default=16)
    parser.add_argument('--mcfl_outer_lr', type=float, default=1e-2)
    parser.add_argument('--mcfl_inner_lr', type=float, default=0.1)
    parser.add_argument('--mcfl_first_order', type=bool, default=True)
    parser.add_argument('--mcfl_recluster_every', type=int, default=1)
    return parser


def get_args():
    return build_parser().parse_args()


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
