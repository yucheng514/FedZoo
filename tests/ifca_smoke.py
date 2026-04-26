from types import SimpleNamespace

import torch

from clients.clientIFCA import IFCAClient
from dataset.ifca_synthetic import make_ifca_synthetic_clients
from models.ifca_models import IFCALinearRegressor
from servers.serverIFCA import IFCAServer


def main():
    args = SimpleNamespace(
        num_clients=8,
        train_frac=0.8,
        ifca_seed=0,
        ifca_clusters=2,
        ifca_synthetic_dim=16,
        ifca_synthetic_samples=40,
        ifca_synthetic_noise=0.01,
        ifca_synthetic_separation=1.0,
    )
    raw_clients, _ = make_ifca_synthetic_clients(args)
    clients = [IFCAClient(client_id=i, data=data, task="regression", device="cpu") for i, data in enumerate(raw_clients)]
    server = IFCAServer(
        cluster_models=[IFCALinearRegressor(input_dim=args.ifca_synthetic_dim) for _ in range(args.ifca_clusters)],
        clients=clients,
        criterion=torch.nn.MSELoss(),
        task="regression",
        device="cpu",
    )

    before = server.evaluate()["test_mse"]
    for _ in range(3):
        server.train_round(lr=0.05, local_epochs=2)
    after = server.evaluate()["test_mse"]

    if after >= before:
        raise SystemExit(f"IFCA smoke test failed: mse did not improve ({before:.4f} -> {after:.4f})")

    print("IFCA smoke test passed.")


if __name__ == "__main__":
    main()
