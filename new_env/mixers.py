import torch
from torch import nn
from torch.nn import functional as F

class VDNMixer(nn.Module): # Value-Decomposition Networks For Cooperative Multi-Agent Learning
    def __init__(self):
        super().__init__()
    
    def process(self, agent_qs):
        n_agents = len(agent_qs)
        batch_size = len(list(agent_qs.values())[0])
        q_vals = torch.zeros(batch_size, n_agents)
        for agent_idx, agent in enumerate(agent_qs):
            for batch_idx, qs in enumerate(agent_qs[agent]):
                q_vals[batch_idx, agent_idx] = qs
        return q_vals

    def forward(self, agent_qs):
        agent_qs = self.process(agent_qs)
        return torch.sum(agent_qs, dim=1)

class QMixer(nn.Module):
    pass