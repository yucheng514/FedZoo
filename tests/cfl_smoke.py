from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

from clients.clientCFL import CFLClient
from models.cfl_models import CFLConvNet
from servers.serverCFL import CFLServer
from utils.mcfl_utils import set_seed


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(device=device)

    x = torch.rand(240, 1, 28, 28)
    y = torch.randint(0, 62, (240,), dtype=torch.long)
    dataset = TensorDataset(x, y)

    client_splits = [Subset(dataset, list(range(0, 120))), Subset(dataset, list(range(120, 240)))]
    model_fn = lambda: CFLConvNet(num_classes=62)
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9)

    clients = [
        CFLClient(args=args, idnum=i, data=data, model_fn=model_fn, optimizer_fn=optimizer_fn, batch_size=32, train_frac=0.8, seed=0)
        for i, data in enumerate(client_splits)
    ]
    server = CFLServer(global_model=model_fn(), device=device)

    server.synchronize_clients(clients)
    for client in clients:
        client.compute_weight_update(epochs=1)
        client.reset()

    similarities = server.compute_pairwise_similarities(clients)
    c1, c2 = server.cluster_clients(similarities)
    cluster_indices = [c1, c2]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices if len(idcs) > 0]
    server.aggregate_clusterwise(client_clusters)
    acc_clients = [client.evaluate() for client in clients]
    server.cache_model(np.arange(len(clients)), clients[0].W, acc_clients)
    print("CFL smoke test passed.")


if __name__ == "__main__":
    main()


