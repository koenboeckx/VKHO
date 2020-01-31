import torch
from torch import nn

params = {
    'board_size':   7,
}

# ----------------------- Agents -----------------------------------------
class Tank:
    def __init__(self, idx):
        super(Tank, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = params['init_ammo']
        self.max_range = 5
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action 
    
    def __repr__(self):
        return self.type + str(self.idx)
    
    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)

class RandomTank(Tank):
    def __init__(self, idx):
        super(RandomTank, self).__init__(idx)
    
    def get_action(self, obs):
        return random.randint(0, 7)

class StaticTank(Tank):
    """Tank that does nothing (always 'do_nothing')"""
    def __init__(self, idx):
        super(StaticTank, self).__init__(idx)
    
    def get_action(self, obs):
        return 0

# ----------------------- Models -----------------------------------------

class MixingNetwork(nn.Module):
    pass

class CommonModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        self.full_in = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.common = nn.Sequential(
            nn.Linear(self.conv_out_size + 64, 128),
            nn.ReLU(),
        )
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Input x has two parts: a 'board' for the conv layer,
        and an 'other', containing ammo and alive flags for
        the fully connecte layer."""
        bs = params['board_size']
        x = x.float()
        board = x[:, :, :, :bs]
        other = x[:, :, 0, bs:bs + 8].view(x.size()[0], -1)
        conv_out = self.conv(board).view(x.size()[0], -1)
        full_out = self.full_in(other)
        common_in  = torch.cat((conv_out, full_out), 1)
        common_out = self.common(common_in)
        return common_out

class AgentNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=128, num_layers=1):
        super().__init__()
        self.common = CommonModel(input_shape)
        self.gru = nn.GRU(128, hidden_size, num_layers=num_layers)
        self.output = nn.Linear(128, n_actions)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def forward(self, x):
        common_out = self.common(x)
        hidden = self.init_hidden(x.size(0))
        _, hidden = self.gru(common_out.unsqueeze(0), hidden)
        values = self.output(hidden) # question: apply ReLU to hidden?
        return values

class QMixAgent:
    def __init__(self):
        self.net = AgentNetwork()
    def sample(self, state, epsilon):
        values = self.net(state)
        if random.sample() < epsilon:
            return 