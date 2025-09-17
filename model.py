import torch.nn as nn
import torch.nn.functional as F


class ReversiNet(nn.Module):
    def __init__(self, channels=128, blocks=12):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([Res(channels) for _ in range(blocks)])
        self.pol = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 65)
        )
        self.val = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            nn.Flatten(),
            nn.Linear(8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.stem(x))
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
