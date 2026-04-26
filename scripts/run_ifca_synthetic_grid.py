import argparse
import itertools
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--clients", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--dims", type=int, nargs="+", default=[100, 500])
    args = parser.parse_args()

    for clusters, clients, samples, dims in itertools.product(args.clusters, args.clients, args.samples, args.dims):
        cmd = [
            sys.executable,
            "main.py",
            "-al",
            "IFCA",
            "-data",
            "IFCA_SYNTHETIC",
            "-gr",
            str(args.rounds),
            "-nc",
            str(clients),
            "--ifca_clusters",
            str(clusters),
            "--ifca_synthetic_samples",
            str(samples),
            "--ifca_synthetic_dim",
            str(dims),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
