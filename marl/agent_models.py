import torch
from torch import nn
from torch import optim
import numpy as np

#---------------------------------- Gym -------------------------------------
class GymModel(nn.Module):
    """Module to be used with open AI gym 'CartPole-v0' """
    def __init__(self, input_shape, n_actions, n_hidden=32, lr=0.01):
        super(GymModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, n_hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(n_hidden, n_actions)
        self.value  = nn.Linear(n_hidden, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-3)
    
    def forward(self, x, dummy): 
        x = self.fc(x)
        value = self.value(x)
        logits = self.policy(x)
        return value, logits

class GymModelIQL(nn.Module):
    """Module to be used with open AI gym 'CartPole-v0' """
    def __init__(self, input_shape, n_actions, n_hidden=32, lr=0.01):
        super(GymModelIQL, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, n_hidden),
            nn.ReLU()
        )
        self.value  = nn.Linear(n_hidden, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-3)
    
    def forward(self, x): 
        x = self.fc(x)
        value = self.value(x)
        return value

#----------------------------------- IQL -----------------------------------
class IQLModel(nn.Module):
    """Defines and learns the behavior of a single agent"""
    def __init__(self, input_shape, n_actions, lr=0.01, board_size=11):
        super(IQLModel, self).__init__()
        if board_size > 10:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
                nn.ReLU(),
            )

        self.conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 128), # reduced hidden layer size # TODO: is 128 large enough for 11x11?
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)   

#------------------------------------ PG -----------------------------------
class PGModel(nn.Module):
    """Defines and learns the behavior of a single agent"""
    def __init__(self, input_shape, n_actions, lr=0.01, board_size=11):
        super(PGModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 128), # TODO: is 128 large enough for 11x11?
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, n_actions) # policy head
        self.value  = nn.Linear(128, 1)         # value head

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x): 
        conv_out = self.conv(x).view(x.size()[0], -1)
        fc_out = self.fc(conv_out)
        value  = self.value(fc_out)
        logits = self.policy(fc_out)
        return value, logits

class PGExtendedModel(nn.Module):
    """Defines and learns the behavior of a single agent
    Extends model above with additional layer for ammo & alive."""
    def __init__(self, input_shape, n_actions, lr=0.01, board_size=11):
        super(PGExtendedModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.in_fc = nn.Sequential( # input fully-connected layer
            nn.Linear(8, 32), # additional input: ammo (4) + alive (4)
            nn.ReLU()
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + 32, 128), # TODO: is 128 large enough for 11x11?
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, n_actions) # policy head
        self.value  = nn.Linear(128, 1)         # value head

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, board, state): # state is here input for in_fc
        conv_out = self.conv(board).view(board.size()[0], -1)
        in_fc_out = self.in_fc(state)
        first_layer_out = torch.cat((conv_out, in_fc_out), 1) # TODO: ? is additional input layer for 'state' needed (m.a.w. feed direct to fc)
        fc_out = self.fc(first_layer_out)
        value = self.value(fc_out)
        log_probs = self.policy(fc_out)
        return value, log_probs

class PG_GRUNet(nn.Module):
    """Defines and learns the behavior of a single agent with a RNN.
    Enables (implicit) conditioning on trajectory in stead of state"""
    def __init__(self, input_shape, hidden_size, n_actions, n_layers=1,
                 lr=0.01, board_size=11):
        super(PG_GRUNet, self).__init__()

        # first Conv2d + fc (ammo, alive), then as input to GRU

        self.hidden_dim = hidden_size
        self.n_layers = n_layers

        # convolutional layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.initial_fc = nn.Sequential( # input fully-connected layer
            nn.Linear(8, 32), # additional input: ammo (4) + alive (4)
            nn.ReLU()
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size + 32, 128), # TODO: is 128 large enough for 11x11?
            nn.ReLU(),
        )
        
        # recurrent layer
        self.rnn = nn.GRU(input_size=128,           # The number of expected features in the input x
                          hidden_size=hidden_size,  # The number of features in the hidden state h
                          num_layers=n_layers,      # Number of recurrent layers (default=1)
                          bias=True,                # Use bias weights
                          )
        # fully-connected layer
        self.policy_head = nn.Linear(self.hidden_dim, n_actions) # policy head
        self.value_head  = nn.Linear(self.hidden_dim, 1)         # value head
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv_layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """ x is tuple (board, other_vars)"""
        board, other = x
        batch_size = board.size(0)

        conv_out  = self.conv_layer(board)
        fc_out    = self.initial_fc(other)
        rnn_input = torch.cat((conv_out, fc_out), 1)

        # initialize hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # passing in the input and hidden state into the model and obtain outputs
        rnn_out, hidden = self.rnn(rnn_input.unsqueeze(0), hidden)

        # reshape the outputs such that they can be fit into fully-connected layer
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        policy = self.policy_head(rnn_out)
        value  = self.value_head(rnn_out)

        return (policy, value), hidden
    
    def init_hidden(self, batch_size):
        """This method generates the first hidden state of zeros which we'll use
        in the forward pass. We'll send the tensor holding the hidden state to
        the device we specified earlier as well."""
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden