import torch
from torch import nn
from torch.nn import functional as F

from env import Observation
from settings import args

class ForwardModel(nn.Module):  
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc3 = nn.Linear(args.n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def forward(self, inputs):
        x = process(inputs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
class RNNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.rnn_hidden_dim = args.n_hidden
        self.fc1 = nn.Linear(input_shape, args.n_hidden)
        self.rnn = nn.GRUCell(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = process(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q

def process(obs_list):
    "transform list of observations into tensor for use in model"
    N = 4 + 3 * len(obs_list[0].friends) + 3 * len(obs_list[0].enemies) # own pos (2), friends alive + pos (3*Nf) , friends alive + pos (3*Ne), ammo (1), aim (1)
    x = torch.zeros((len(obs_list), N)) 
    if isinstance(obs_list[0], Observation):
        for obs_idx, obs in enumerate(obs_list):
            x[obs_idx, 0:2] = torch.tensor(obs.own_position)
            idx = 2
            for friend in obs.friends:
                if friend:  # friend is alive
                    x[obs_idx, idx:idx+3] = torch.tensor([1.,] + list(friend))
                else:       # friend is dead
                    x[obs_idx, idx:idx+3] = torch.tensor([0., 0., 0.])
                idx += 3
            for enemy in obs.enemies:
                if enemy:   # enemy is alive
                    x[obs_idx, idx:idx+3] = torch.tensor([1.,] + list(enemy))
                else:       # enemy is dead
                    x[obs_idx, idx:idx+3] = torch.tensor([0., 0., 0.])
                idx += 1
            x[obs_idx, idx]   = obs.ammo / args.init_ammo
            x[obs_idx, idx+1] = obs.aim.id if obs.aim is not None else -1
    else:
        raise ValueError(f"x should be (list of) Observation(s)")
    return x