from . import agent_models
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents 

class PGAgent(agents.BaseAgent):
    """
    Agent class to be used with policy gradient RL.
    Model is set with method .set_model()
    """
    def __init__(self):
        super(PGAgent, self).__init__()