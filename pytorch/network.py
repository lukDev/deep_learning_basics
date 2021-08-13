import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.model(x)
