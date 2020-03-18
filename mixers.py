import torch
from torch import nn
from torch.nn import functional as F

class VDNMixer(nn.Module): # Ref: Value-Decomposition Networks For Cooperative Multi-Agent Learning
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs, states):
        return torch.sum(agent_qs, dim=1).unsqueeze(-1)

class QMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Hypernetwork
        self.n_trainers = args.n_friends # assumes all friends are learning
        self.embed_dim  = args.embed_dim
        self.state_dim = 5 * args.n_agents # every agent is represented by 5 values: x, y, alive, ammo, aim
        self.HW1 = nn.Linear(self.state_dim, self.embed_dim * self.n_trainers)
        self.Hb1 = nn.Linear(self.state_dim, self.embed_dim)
        self.HW2 = nn.Linear(self.state_dim, self.embed_dim)
        self.Hb2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        self.qmix_ns = args.qmix_ns # no conditioning on state information

    def forward(self, agent_qs, states):
        # computes matrices via hypernetwork
        agent_qs = agent_qs.unsqueeze(-1) # add dimension for torch.bmm
        states = states.reshape(-1, self.state_dim)
        if self.qmix_ns:
            states = torch.zeros_like(states) # only for testing
        
        W1 = self.HW1(states)
        #W1 = W1 - W1.mean(dim=1).unsqueeze(1) # normalize
        W1 = torch.abs(W1).reshape(-1, self.embed_dim, self.n_trainers)
        
        b1 = self.Hb1(states)
        b1 = b1.reshape(-1, self.embed_dim, 1)
        
        W2 = self.HW2(states)
        #W2 = W2 - W2.mean(dim=1).unsqueeze(1) # normalize
        W2 = torch.abs(W2).reshape(-1, 1, self.embed_dim)
        
        b2 = self.Hb2(states)
        b2 = b2.reshape(-1, 1, 1)

        # real network updates          # agent_qs = (bs x n_trainers)
        QW1 = torch.bmm(W1, agent_qs)   # (bs x embed_dim x 1)
        Qb1 = F.elu(QW1 + b1)           # (bs x embed_dim x 1)
        QW2 = torch.bmm(W2, Qb1)        # (bs x 1 x 1)
        Qtot = QW2 + b2                 # (bs x 1 x 1)
        return Qtot.squeeze(-1)         # (bs x 1)
