# FedZoo

This repository now contains both the original FedAvg pipeline and an integrated minimal MCFL pipeline.

## Algorithms

- `FedAvg` (existing implementation)
- `MCFL` (integrated from `mcfl_minimal_project` without overwriting original modules)

## MCFL module mapping

- `clients/clientMCFL.py`
- `servers/serverMCFL.py`
- `models/mcfl_models.py`
- `dataset/mcfl_synthetic.py`
- `utils/mcfl_utils.py`
- `utils/mcfl_clustering.py`

## Run examples

```bash
python main.py -al FedAvg
python main.py -al MCFL -gr 10 -nc 12 -ncl 2 --mcfl_num_clusters 3
```
