import argparse
import subprocess
import sys


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the checks.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "main", "all"],
        default="all",
        help="Which IFCA pretest to run.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Rounds for the main-entry synthetic run.")
    parser.add_argument("--clients", type=int, default=8, help="Number of synthetic clients.")
    parser.add_argument("--clusters", type=int, default=2, help="Number of IFCA clusters.")
    parser.add_argument("--samples", type=int, default=40, help="Synthetic samples per client.")
    parser.add_argument("--dims", type=int, default=16, help="Synthetic feature dimension.")
    args = parser.parse_args()

    if args.mode in {"smoke", "all"}:
        run_command([args.python, "-m", "tests.ifca_smoke"])

    if args.mode in {"main", "all"}:
        run_command(
            [
                args.python,
                "main.py",
                "-al",
                "IFCA",
                "-data",
                "IFCA_SYNTHETIC",
                "-gr",
                str(args.rounds),
                "-nc",
                str(args.clients),
                "--ifca_clusters",
                str(args.clusters),
                "--ifca_synthetic_samples",
                str(args.samples),
                "--ifca_synthetic_dim",
                str(args.dims),
                "--ifca_seed",
                "0",
                "--ifca_tau",
                "2",
                "-lr",
                "0.05",
            ]
        )


if __name__ == "__main__":
    main()
