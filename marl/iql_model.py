import torch
from torch import nn

class IQL(nn.Module):
    def __init__(self):
        super(IQL, self).__init__()
    
    def forward(self, x):
        return