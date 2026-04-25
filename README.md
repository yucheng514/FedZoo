# FedZoo

This repository now contains both the original FedAvg pipeline and an integrated minimal MCFL pipeline.

## Algorithms

- `FedAvg` (existing implementation)
- `MCFL` (formalized version with multi-batch local adaptation, real-data fallback, and stronger backbone selection)

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
python main.py -al MCFL -gr 10 -nc 20 -ncl 10 --mcfl_backbone auto
```
