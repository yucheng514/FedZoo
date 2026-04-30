from types import SimpleNamespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.mcfl_synthetic import make_mcfl_clients
from models.mcfl_models import MCFLMLPClassifier
from servers.serverMCFL import MCFLServer
from utils.mcfl_utils import set_seed


def _mean_adapted_accuracy(clients, server):
    return sum(
        client.evaluate(server.cluster_models[client.cluster_id], adapt=True, inner_lr=0.05, local_epochs=1)
        for client in clients
    ) / len(clients)


def main():
    set_seed(0)
    args = SimpleNamespace(
        dataset="synthetic",
        device="cpu",
        num_clients=8,
        num_classes=3,
        batch_size=8,
        local_epochs=1,
        few_shot=0,
        mcfl_seed=0,
        mcfl_backbone="mlp",
        mcfl_input_dim=16,
        mcfl_samples_per_client=64,
        mcfl_true_groups=2,
        mcfl_hidden_dim=32,
        mcfl_support_ratio=0.75,
    )

    for cluster_method in ("kmeans", "agglomerative"):
        clients = make_mcfl_clients(args)
        model = MCFLMLPClassifier(in_dim=args.mcfl_input_dim, hidden_dim=args.mcfl_hidden_dim, num_classes=args.num_classes)
        server = MCFLServer(
            global_model=model,
            num_clusters=2,
            encoder_embed_dim=8,
            outer_lr=5e-2,
            model_mix=0.5,
            device="cpu",
            total_rounds=3,
            recluster_every=1,
            recluster_warmup_rounds=0,
            stop_recluster_after=-1,
            skip_final_recluster=True,
            cluster_method=cluster_method,
            cluster_feature="head_updates",
        )
        server.assign_initial_clusters(clients)

        before = _mean_adapted_accuracy(clients, server)
        before_params = [torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone() for model in server.cluster_models]

        for round_idx in range(3):
            stats = server.train_round(clients, round_idx=round_idx, inner_lr=0.05, first_order=True, local_epochs=1)
            if not stats:
                raise SystemExit(f"MCFL smoke test failed ({cluster_method}): train_round returned no client stats.")
            for stat in stats:
                for key in ("support_loss", "query_loss", "support_acc", "query_acc"):
                    value = float(stat[key])
                    if not torch.isfinite(torch.tensor(value)):
                        raise SystemExit(
                            f"MCFL smoke test failed ({cluster_method}): non-finite {key}={value} in round {round_idx}."
                        )

        after = _mean_adapted_accuracy(clients, server)
        after_params = [torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone() for model in server.cluster_models]

        if not torch.isfinite(torch.tensor(before)) or not torch.isfinite(torch.tensor(after)):
            raise SystemExit(
                f"MCFL smoke test failed ({cluster_method}): non-finite adapted accuracy ({before:.4f} -> {after:.4f})"
            )

        total_delta = sum((after_vec - before_vec).norm().item() for before_vec, after_vec in zip(before_params, after_params))
        if total_delta <= 0:
            raise SystemExit(f"MCFL smoke test failed ({cluster_method}): cluster model parameters did not change after training.")

    print("MCFL smoke test passed.")


if __name__ == "__main__":
    main()
