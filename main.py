import argparse
import os
import time
import torch
from src.models.models import FedAvgCNN
from src.servers.serverAvg import FedAvg
import copy
from torch import nn

torch.manual_seed(0)


def run(args):
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
    
        # select algorithm
        if args.algorithm == "FedAvg":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-al', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-dev', "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Multiple update steps in one local epoch.")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Total number of clients")


    args = parser.parse_args()
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
        # print("\nCUDA is not available, falling back by priority: mps -> cpu.\n")
        args.device = "mps" if mps_available else "cpu"
    elif args.device == "mps" and not mps_available:
        # print("\nMPS is not available, falling back by priority: cuda -> cpu.\n")
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {args.device}\n")
    print("=== args " + "=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=== args end " + "=" * 46)
    run(args)