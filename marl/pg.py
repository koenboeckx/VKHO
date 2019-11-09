from . import agent_models
from .common import *
from tensorboardX import SummaryWriter

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn



class PGAgent(agents.Tank):
    """
    Agent class to be used with policy gradient RL.
    Model is set with method .set_model()
    """
    def __init__(self, idx, board_size=11):
        super(PGAgent, self).__init__(idx)
        self.init_agent(idx)
        self.board_size = board_size
    
    def set_model(self, input_shape, n_actions, lr, device):
        """
        Set the model (neural net) of the agent.

        :param input_shape: shape of input vector (channels, widht, height)
        :param lr:          learning rate for model optimizer (=Adam)
        :param device:      torch.device("cpu" or "cuda")
        :return: None
        """
        self.model = agent_models.PGModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(device)
    
    def get_action(self, state, device):
        state_v = preprocess([state]).to(device)
        _, logprobs = self.model(state_v)
        probs = nn.Softmax(dim=1)(logprobs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()