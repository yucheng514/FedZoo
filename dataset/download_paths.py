from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DATA_ROOT = DATASET_DIR / "data"


def dataset_data_root(name):
    path = DATA_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_torchvision_root(configured_root, dataset_name):
    if configured_root and configured_root not in {".", "./", "dataset/data"}:
        path = Path(configured_root).expanduser().resolve()
    else:
        path = dataset_data_root(dataset_name)
    path.mkdir(parents=True, exist_ok=True)
    return path
