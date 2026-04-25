import torch
import torch.nn.functional as F


class CFLConvNet(torch.nn.Module):
    def __init__(self, num_classes=62, in_channels=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

