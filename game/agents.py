"""
Contains all types of agents available
"""

class BaseAgent:
    """
    This is the base abstraction for agents. 
    All agents should inherit from this class
    """

    def __init__(self):
        pass

    def __getattr__(self, attr):
        return getattr(self, attr)
    
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

    def init_agent(self, id, game_type):
        self.id = id