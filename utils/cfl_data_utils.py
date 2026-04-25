import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Subset


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    """Split indices according to a Dirichlet label distribution."""
    train_idcs = np.asarray(train_idcs)
    train_labels = np.asarray(train_labels)
    n_classes = int(train_labels.max()) + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
    return client_idcs


class CustomSubset(Subset):
    """Subset with an optional transform applied at retrieval time."""

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.subset_transform:
            x = self.subset_transform(x)
        if not torch.is_tensor(x):
            x = to_tensor(x)
        return x, y

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

