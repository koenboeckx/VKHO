"""Independent Actor-Critic
from https://arxiv.org/abs/1705.08926
"""

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class IACModel(nn.Module):
    """Simple recurrent ANN consisting of:
        * input  : fully-connected layer (inputshape x args.rnn_hidden_dim)
        * RNN    : GRU layer
        * output : fully-connected layer (args.rnn_hidden_dim x args.n_actions)
    """
    def __init__(self, input_shape, args):
        super(IACModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

     def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = self._process_input(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(x)
        return q, h

class IACAgent(agents.Tank):
