# FedZoo

MCFL：0.7844 ± 0.0193
seed 值：[0.7824, 0.7706, 0.8181, 0.7775, 0.7735]
CFL：0.7289 ± 0.0088
seed 值：[0.7273, 0.7414, 0.7196, 0.7339, 0.7225]

This repository now contains the original FedAvg pipeline plus integrated MCFL, CFL, IFCA, Per-FedAvg, and pFedMe pipelines.

## Algorithms

- `FedAvg` (existing implementation)
- `MCFL` (formalized version with multi-batch local adaptation, real-data fallback, and stronger backbone selection)
- `CFL` (clustered federated learning extracted from the notebook into reusable runtime modules)
- `IFCA` (iterative federated clustering integrated from the legacy experiment code)
- `PerFedAvg` / `PerAvg` (personalized federated learning with one-step evaluation)
- `pFedMe` (personalized federated learning with Moreau envelope regularization)

## MCFL module mapping

- `clients/clientMCFL.py`
- `servers/serverMCFL.py`
- `models/mcfl_models.py`
- `dataset/mcfl_synthetic.py`
- `utils/mcfl_utils.py`
- `utils/mcfl_clustering.py`

## CFL module mapping

- `clients/clientCFL.py`
- `servers/serverCFL.py`
- `models/cfl_models.py`
- `dataset/cfl_emnist.py`
- `utils/cfl_data_utils.py`
- `utils/cfl_federation.py`
- `utils/cfl_helper.py`

## IFCA module mapping

- `clients/clientIFCA.py`
- `servers/serverIFCA.py`
- `models/ifca_models.py`
- `dataset/ifca_synthetic.py`
- `dataset/ifca_rotated_mnist.py`
- `dataset/ifca_rotated_cifar.py`
- `dataset/ifca_emnist.py`

`IFCA` now covers the main legacy experiment modes directly in the primary runtime:

- `--ifca_mode clustered`: iterative reassignment each round
- `--ifca_mode oneshot`: assign clients once, then keep assignments fixed
- `--ifca_mode local`: one model per client
- `--ifca_init_rounds N`: warm-start cluster models with fixed assignments
- `--ifca_freeze_backbone`: train only the classifier head when the model exposes one

## Per-FedAvg and pFedMe module mapping

### Per-FedAvg (PerFedAvg / PerAvg)

- `clients/clientPerFedAvg.py`
- `servers/serverPerFedAvg.py`
- `utils/pfedme_optimizer.py` (PerAvgOptimizer)

Personalized federated learning with two-step training and one-step evaluation.

### pFedMe

- `clients/clientpFedMe.py`
- `servers/serverpFedMe.py`
- `utils/pfedme_optimizer.py` (pFedMeOptimizer)

Personalized federated learning with Moreau envelope regularization and dual-track parameters.

Key parameters:
- `--beta`: interpolation factor (default 0.5)
- `--lamda`: regularization strength (default 15.0)
- `--K`: personalized optimization steps (default 5)
- `--p_learning_rate`: personalized learning rate (default 0.01)

See `INTEGRATION_PERFEDAVG_PFEDME.md` for detailed documentation.



## Run examples

```bash
# FedAvg
python main.py -al FedAvg -data CIFAR10 -gr 100 -nc 20 -ncl 10 --log_file logs/fedavg_cifar10.log
python main.py -al FedAvg -data CIFAR10 -gr 100 -nc 18 -nnc 2 -ncl 10 --eval_new_clients --fine_tuning_epoch_new 5 --log_file logs/fedavg_ft_cifar10.log

# MCFL
python main.py -al MCFL -data CIFAR10 -gr 100 -nc 20 -ncl 10 --mcfl_backbone auto --log_file logs/mcfl_cifar10.log
python main.py -al MCFL -data CIFAR10 -gr 10 --log_file logs/mcfl_seed1.txt

# CFL
python main.py -al CFL -gr 80 -nc 10 -ncl 62 --cfl_dirichlet_alpha 1.0

# IFCA
python main.py -al IFCA -data IFCA_SYNTHETIC -gr 50 -nc 20 --ifca_clusters 2 --ifca_synthetic_dim 100
python main.py -al IFCA -data MNIST -gr 30 -nc 20 -ncl 10 --ifca_clusters 4 --ifca_tau 5
python main.py -al IFCA -data CIFAR10 -gr 20 -nc 20 -ncl 10 --ifca_clusters 4 --ifca_mode clustered
python main.py -al IFCA -data FEMNIST -gr 20 -nc 20 -ncl 62 --ifca_clusters 4 --ifca_mode oneshot --ifca_freeze_backbone

# Per-FedAvg
python main.py -al PerFedAvg -m CNN -data MNIST -gr 100 -nc 20 -lbs 10 -ls 5 -lr 0.01 --log_file logs/perfedavg_mnist.log
python main.py -al PerFedAvg -m CNN -data CIFAR10 -gr 100 -nc 20 -lbs 32 -ls 5 -lr 0.01 --log_file logs/perfedavg_cifar10.log

# pFedMe
python main.py -al pFedMe -m CNN -data MNIST -gr 100 -nc 20 -lbs 10 -ls 5 -lr 0.005 --beta 0.5 --lamda 15.0 --K 5 --p_learning_rate 0.01 --log_file logs/pfedme_mnist.log
python main.py -al pFedMe -m CNN -data CIFAR10 -gr 100 -nc 20 -lbs 32 -ls 5 -lr 0.005 --beta 0.5 --lamda 15.0 --K 5 --p_learning_rate 0.01 --log_file logs/pfedme_cifar10.log
```

## IFCA experiment scripts

- `python scripts/run_ifca_synthetic_grid.py`
- `python scripts/run_ifca_image_modes.py --dataset MNIST`
