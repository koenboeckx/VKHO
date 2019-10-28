import torch
from torch import nn
from torch import optim
import numpy as np

class IQL(nn.Module):
    """Defines and learns the behavior of a single agent"""
    def __init__(self, input_shape, n_actions, lr=0.01, board_size=11):
        super(IQL, self).__init__()
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
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optim = optim.Adam(self.parameters(), lr=lr)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)    