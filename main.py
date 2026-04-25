import time
import torch
from models.models import FedAvgCNN
from servers.serverAvg import FedAvg
import copy
from torch import nn
from config import get_args, resolve_device

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

        server.train()

        time_list.append(time.time() - start)


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