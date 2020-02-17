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
    
    def forward(self, x):
        x = self.process(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def process(self, obs_list):
        "transform list of observations into tensor"
        x = torch.zeros((len(obs_list), 13)) # own pos (2), friends alive + pos (3) , friends alive + pos (2x3), ammo (1), aim (1)
        if isinstance(obs_list[0], Observation):
            for obs_idx, obs in enumerate(obs_list):
                x[obs_idx, 0:2] = torch.tensor(obs.own_position)
                friend_pos = obs.friends[0] # TODO: improve below to allow more friends/enemies
                if friend_pos:     
                    x[obs_idx, 2:5] = torch.tensor([1.,] + list(obs.friends[0]))
                else:
                    x[obs_idx, 2:5] = torch.tensor([0., 0., 0.])
                enemy1_pos = obs.enemies[0]
                if enemy1_pos:
                    x[obs_idx, 5:8] = torch.tensor([1.,] + list(obs.enemies[0]))
                else:
                    x[obs_idx, 5:8] = torch.tensor([0., 0., 0.])
                enemy2_pos = obs.enemies[1]
                if enemy2_pos:
                    x[obs_idx, 8:11] = torch.tensor([1.,] + list(obs.enemies[1]))
                else:
                    x[obs_idx, 8:11] = torch.tensor([0., 0., 0.])

                x[obs_idx, 11]   = obs.ammo / args.init_ammo
                x[obs_idx, 12]   = int(obs.aim is not None)
        else:
            raise ValueError(f"x should be (list of) Observation(s)")
        return x

class RNNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.rnn_hidden_dim = n_hidden
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
    "transform list of observations into tensor"
    x = torch.zeros((len(obs_list), 13)) # own pos (2), friends alive + pos (3) , friends alive + pos (2x3), ammo (1), aim (1)
    if isinstance(obs_list[0], Observation):
        for obs_idx, obs in enumerate(obs_list):
            x[obs_idx, 0:2] = torch.tensor(obs.own_position)
            friend_pos = obs.friends[0] # TODO: improve below to allow more friends/enemies
            if friend_pos:     
                x[obs_idx, 2:5] = torch.tensor([1.,] + list(obs.friends[0]))
            else:
                x[obs_idx, 2:5] = torch.tensor([0., 0., 0.])
            enemy1_pos = obs.enemies[0]
            if enemy1_pos:
                x[obs_idx, 5:8] = torch.tensor([1.,] + list(obs.enemies[0]))
            else:
                x[obs_idx, 5:8] = torch.tensor([0., 0., 0.])
            enemy2_pos = obs.enemies[1]
            if enemy2_pos:
                x[obs_idx, 8:11] = torch.tensor([1.,] + list(obs.enemies[1]))
            else:
                x[obs_idx, 8:11] = torch.tensor([0., 0., 0.])

            x[obs_idx, 11]   = obs.ammo / args.init_ammo
            x[obs_idx, 12]   = int(obs.aim is not None)
    else:
        raise ValueError(f"x should be (list of) Observation(s)")
    return x