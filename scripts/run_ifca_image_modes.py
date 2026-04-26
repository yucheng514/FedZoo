import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10", "FEMNIST"], default="MNIST")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=["clustered", "oneshot", "local"])
    args = parser.parse_args()

    for mode in args.modes:
        cmd = [
            sys.executable,
            "main.py",
            "-al",
            "IFCA",
            "-data",
            args.dataset,
            "-gr",
            str(args.rounds),
            "-nc",
            str(args.clients),
            "-ncl",
            "62" if args.dataset == "FEMNIST" else "10",
            "--ifca_clusters",
            str(args.clusters),
            "--ifca_mode",
            mode,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
