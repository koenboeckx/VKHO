"""
Implement Counterfactual Multi-Agent Policy Gradients
https://arxiv.org/abs/1705.08926
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 

class Actor(nn.Module):
    def __init__(self, input_shape, gru_input_size, hidden_size, n_actions):
        super(Actor, self).__init__()

        self.input_fc = nn.Sequential( # takes as input: (observation of agent i,
            nn.Linear(input_shape[0], gru_input_size), #  advantage a,  
            nn.ReLU()                                  #  previous actions u_t-1)
        )
        self.rnn = nn.GRU(gru_input_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, n_actions)
    
    def _init_hidden(self):
        pass

    def foward(self, x):
        x = self.input_fc(x)
        pass