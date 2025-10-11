import torch.nn as nn
import torch.nn.functional as F


class ReversiNet(nn.Module):
    def __init__(self, channels=128, blocks=12):
        super().__init__()
        # Larger network: 128 channels, 12 blocks (up from 64 channels, 6 blocks)
        # Input: 5 channels (black, white, side, corners, edges)
        self.stem = nn.Conv2d(5, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([Res(channels) for _ in range(blocks)])

        # Enhanced policy head with batch norm
        self.pol = nn.Sequential(
            nn.Conv2d(channels, 4, 1),  # Increased from 2 to 4 channels
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 65)
        )

        # Enhanced value head with more capacity
        self.val = nn.Sequential(
            nn.Conv2d(channels, 2, 1),  # Increased from 1 to 2 channels
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 128),  # Increased hidden size
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return self.pol(x), self.val(x)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        self.b1, self.b2 = nn.BatchNorm2d(c), nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)
