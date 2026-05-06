import torch
import torch.nn as nn
import torch.nn.functional as F


class MCFLFormalMLP(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=256, num_classes=2, dropout=0.2):
        super().__init__()
        first_layer = nn.LazyLinear(hidden_dim) if in_dim is None else nn.Linear(in_dim, hidden_dim)
        self.net = nn.Sequential(
            first_layer,
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
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


class MCFLMLPClassifier(MCFLFormalMLP):
    pass


class MCFLClientEncoder(nn.Module):
    """改进 5: 增强的编码器 - 更好的聚类特征表示"""

    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        hidden_dim = 256

        # 改进: 添加批归一化, 更深的网络, 更好的非线性
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, embed_dim),
            nn.BatchNorm1d(embed_dim),  # 最后一层也添加BN
        )

    def forward(self, update_vec):
        z = self.net(update_vec)
        # L2 归一化用于聚类
        return F.normalize(z, p=2, dim=-1)

