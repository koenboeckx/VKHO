import torch
from torch import nn
from torch.nn import functional as F

from env import Observation
from settings import params

class PGModel(nn.Module):   # Q-Learning Model
    def __init__(self, input_shape, n_hidden, n_actions, lr):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
    
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

                x[obs_idx, 11]   = obs.ammo / params['init_ammo']
                x[obs_idx, 12]   = int(obs.aim is not None)
        else:
            raise ValueError(f"x should be (list of) Observation(s)")
        return x

class IQLModel(nn.Module):   # Q-Learning Model
    def __init__(self, input_shape, n_hidden, n_actions, lr):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
    
    def forward(self, x):
        x = self.process(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        vals = self.fc3(x)
        return vals
    
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

                x[obs_idx, 11]   = obs.ammo / params['init_ammo']
                x[obs_idx, 12]   = int(obs.aim is not None)
        else:
            raise ValueError((f"x should be (list of) Observation(s)"))
        return x
