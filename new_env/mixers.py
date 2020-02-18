import torch
from torch import nn
from torch.nn import functional as from

class VDNMixer(nn.Module): # Value-Decomposition Networks For Cooperative Multi-Agent Learning
    def __init__(self):
        super().__init__()
    
    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=2, keepdim=True)