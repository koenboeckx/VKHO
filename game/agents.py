"""
Contains all types of agents available
"""

import random   # for RandomTank()

class BaseAgent:
    """
    This is the base abstraction for agents. 
    All agents should inherit from this class
    """

    def __init__(self):
        pass

#    def __getattr__(self, attr):
#        return getattr(self, attr)
    
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

#    def init_agent(self, id_, game_type):
#        self.id = id_
    
    def __repr__(self):
        return self.type + str(self.idx)

class TestAgent(BaseAgent):
    def __init__(self, idx):
        super(TestAgent, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'x'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = 5
        self.pos = None # initialized by environment    

class Tank(BaseAgent):
    def __init__(self, idx):
        super(Tank, self).__init__()
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

class RandomTank(Tank):
    def __init__(self, idx):
        super(RandomTank, self).__init__(idx)
    
    def get_action(self, obs):
        return random.randint(0, 7)
