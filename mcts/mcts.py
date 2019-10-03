"""
Attempt to solve - or find non-trivial tactics - for the game
For now: only symmetric game with 2 Tanks per Player (=Commander)
"""

from itertools import product   # create joint action spaces

import game
from game.agents import Tank

def obs_to_int(obs):
    """convert an observation to a unique integer. Here, an observation
    if the combined observation of the two agents on one team"""
    return hash(str(obs))

class CommandedTank(Tank):
    """Tank that gets actions from a commander"""
    def __init__(self, idx, commander):
        super(CommandedTank, self).__init__(idx)
        self.commander = commander
        self.commander.add_agent(self)
    
    def get_action(self, obs):
        """Asks the commander to provide an action"""
        return self.commander.get_action(obs) # TODO: is this needed?

class MCTSPlayer:
    """Player that commands 2 tanks. Uses MCTS to search action space
    and provide best action for both tanks when requested."""
    def __init__(self):
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)
    
    def set_environment(self, env):
        self.env = env
        self.action_space = list(product(env.action_space.keys(),
                                         env.action_space.keys()))
    
    def get_actions(self, obs):
        obs_int = obs_to_int(obs)
        if obs_int in self.n_visits:
            self.n_visits[obs_int] += 1
        else:
            self.n_visits[obs_int] = 1
        return (4, 4)
    
    def init_stores(self):
        self.n_visits = {}




