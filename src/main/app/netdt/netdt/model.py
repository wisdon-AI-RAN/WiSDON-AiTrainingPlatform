import torch
import torch.nn as nn


class AERegressor(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        self.dec = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        y = self.dec(z)
        return y, z