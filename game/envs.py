"""
v0 of the game Environment.
Limitations:
    * Fully observable, no terrain
    * Fixed size board (11x11)
    * Fixed teams: 2 against 2
"""

import random
from collections import namedtuple

from . import agents

# helper function
def flatten(board):
    """Flatten the 'board' dictionary
    :params:    board: dictionary containing different elements
    :return:    flattened_board: 121 Ints
    """
    flattened_board = []
    for i in range(11):
        for j in range(11):
            if board[(i,j)] is None:
                flattened_board.append(-1)
            else:
                flattened_board.append(board[(i,j)].index) # TODO: define agent index ?
    return flattened_board              


# Observation: what does each agent know/see?
Observation = namedtuple('Observation', [
    'board',        # flattened board - 121 Ints
    'position',     # position of agent in grid - 2 Ints in [0, 10]
    'alive',        # is agent alive? - Int in [0,1]
    'ammo',         # current ammo level
    'team_mate',    # which agent is teammate - 1 Int in [0,3]
    'enemies'       # which agents are this agent's enemies - 2 Ints in [0, 3]
])

class Environment:
    """The base game environment."""
    def __init__(self, **kwargs):
        self.agents = []
        self.n_agents = 0
        self.args = kwargs
        self.state = None

        self.set_init_game_state()
    
    def add_agent(self, id_, agent):
        assert isinstance(agent, agents.BaseAgent)
        self.agents.append(agent)
        self.n_agents += 1
    
    def set_init_game_state(self):
        """Set the initial game state.
        A game state consists of a list of agents states.
        Creates the internal state of the environment.

        :returns: a list of observations for the 4 agents
        """
        self.board_size = self.args.get('size', 11) # board size fixed on 11x11
        self.board = {}
        for i in range(self.board_size):
           for j in range(self.board_size):
               self.board[(i,j)] = None
        
        # Position players randomly on the board
        for agent in self.agents:
            i, j = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            while self.board[(i,j)] is not None:
                i, j = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            agent.pos = (i, j)
            self.board[(i,j)] = agent
        
        # generate the observations for the 4 players
        # by default: player 0 & 1 are 1 team, players 2 & 3 are the other team
        obs = []
        for agent_idx, agent in self.agents:
            if agent_idx in [0, 1]:
                team_mate = 1-agent_idx
                enemies = [2, 3]
            else:
                team_mate = 3 if agent_idx == 2 else 2
                enemies = [0, 1]

            observation = Observation(
                'board' = flatten(self.board),
                'position' = agent.pos,
                'alive' = 1,
                'ammo' = 5,
                'team_mate' = team_mate,
                'enemies' = enemies
            )
            obs.append(observation)
        return obs


    def render(self):
        """Represent the state of the environment"""
        board_repr = ''
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[(i,j)] == None:
                   board_repr += ' ' * 4
                else:
                    board_repr += ' ' + repr(self.board[(i,j)]) + ' '
            board_repr += '\n'
        
        print(board_repr)

    def act(self, observations):
        """Return the chosen action for eac agent,
        based on the global observation.
        :params:
            observations: tuple oof individual observations
        :returns:
            list of actions, one for each agent in agent_list
        """
        actions = []
        for obs, agent in zip(observations, self.agents):
            actions.append(agent.get_action(obs))
        return actions

    def step(self, actions):
        """Perform actions, part of joint action space.
        Deconflict simultanuous execution of actions (...)
        """
        for agent, action in zip(self.agents, actions):
            pass # change environemnt state based on agent action
