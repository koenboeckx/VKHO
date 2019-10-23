"""
Independent Q-Learning.
"""

import random
from . import iql_model

class BaseAgent:
    """
    This is the base abstraction for agents. 
    All agents should inherit from this class
    """

    def __init__(self):
        pass
    
    def get_action(self, obs, action_space):
        """Return action to be executed by environment"""
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent
        know the episode has ended and what the reward is
        
        Args:
            reward: the single reward scalar to this agent.
        """
        pass

    def __repr__(self):
        return self.type + str(self.idx)

class IQLAgent(BaseAgent):
    def __init__(self, idx):
        super(IQLAgent, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = 5
        self.max_range = 4
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action
        self.set_model()
    
    def set_model(self):
        agent.model = iql_model.IQL((1,8,8), 8)

    def get_action(self, obs):
        return random.randint(0, 7)  

def train(env, agent, n_steps=10, epsilon=1.0):
    if not hasattr(agent, 'model'):
        input_shape = (1, env.board_size, env.board_size)
        model = iql_model.IQL(input_shape, env.action_space_n)
        agent.model = model
    buffer = []
    state = env.set_init_game_state()
    for step in range(n_steps):
        action = eps_greedy()


