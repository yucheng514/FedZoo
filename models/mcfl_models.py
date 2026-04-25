import torch.nn as nn
import torch.nn.functional as F


class MCFLFormalMLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MCFLMLPClassifier(MCFLFormalMLP):
    pass


class MCFLClientEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, update_vec):
        z = self.net(update_vec)
        return F.normalize(z, dim=-1)

