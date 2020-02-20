import torch
from torch import nn
from torch.nn import functional as F

from settings import args

def process_qs(agent_qs):
    """Transforms Qs into tensor to be used in network."""
    n_agents = len(agent_qs)
    batch_size = len(list(agent_qs.values())[0])
    q_vals = torch.zeros(batch_size, n_agents)
    for agent_idx, agent in enumerate(agent_qs):
        for batch_idx, qs in enumerate(agent_qs[agent]):
            q_vals[batch_idx, agent_idx] = qs
    return q_vals

def process_states(states):
    """Transforms State into tensor to be used in network."""
    batch_size = len(states)
    n_agents = len(states[0].agents)
    states_v = torch.zeros(batch_size, n_agents, 5)
    for state_idx, state in enumerate(states):
        for agent_idx, agent in enumerate(state.agents):
            states_v[state_idx, agent_idx, 0] = state.position[agent][0] # x
            states_v[state_idx, agent_idx, 1] = state.position[agent][1] # y
            states_v[state_idx, agent_idx, 2] = state.alive[agent]
            states_v[state_idx, agent_idx, 3] = state.ammo[agent] / args.init_ammo
            states_v[state_idx, agent_idx, 4] = -1 if state.aim[agent] is None else state.aim[agent].id
    return states_v

class VDNMixer(nn.Module): # Value-Decomposition Networks For Cooperative Multi-Agent Learning
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs, states):
        agent_qs = process_qs(agent_qs)
        return torch.sum(agent_qs, dim=1)

class QMixer(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Hypernetwork
        self.n_trainers = args.n_friends + 1
        self.embed_dim = embed_dim
        self.state_dim = 5*args.n_agents
        self.HW1 = nn.Linear(self.state_dim, embed_dim * self.n_trainers)
        self.Hb1 = nn.Linear(self.state_dim, embed_dim)
        self.HW2 = nn.Linear(self.state_dim, embed_dim)
        self.Hb2 = nn.Linear(self.state_dim, 1)

    def forward(self, agent_qs, states):
        agent_qs = process_qs(agent_qs)
        states = process_states(states)

        # computes matrices via hypernetwork
        states = states.reshape(-1, self.state_dim)
        W1 = torch.abs(self.HW1(states))
        W1 = W1.reshape(-1, self.embed_dim, self.n_trainers)
        b1 = self.Hb1(states)
        W2 = torch.abs(self.HW2(states))
        W2 = W2.reshape(-1, 1, self.embed_dim)
        b2 = F.relu(self.Hb2(states))

        # real network updates
        QW1 = torch.bmm(W1, agent_qs.unsqueeze(2)).squeeze()
        Qb1 = F.elu(b1 + QW1)
        QW2 = torch.bmm(W2, Qb1.unsqueeze(2)).squeeze()
        Qtot = QW2 + b2
        return Qtot