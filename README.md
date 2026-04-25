# FedZoo

This repository now contains the original FedAvg pipeline plus integrated MCFL and CFL pipelines.

## Algorithms

- `FedAvg` (existing implementation)
- `MCFL` (formalized version with multi-batch local adaptation, real-data fallback, and stronger backbone selection)
- `CFL` (clustered federated learning extracted from the notebook into reusable runtime modules)

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



## Run examples

```bash
python main.py -al FedAvg
python main.py -al MCFL -gr 10 -nc 20 -ncl 10 --mcfl_backbone auto
python main.py -al CFL -gr 80 --cfl_num_clients 10 --cfl_num_classes 62 --cfl_dirichlet_alpha 1.0
```
