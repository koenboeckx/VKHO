"""
v0 of the game Environment.
Limitations:
    * Fully observable, no terrain
    * Fixed size board (11x11)
    * Fixed teams: 2 against 2
"""

all_actions = { 0: 'do_nothing',
                1: 'aim1',  # prepare to fire on first  enemy (0 or 2)
                2: 'aim2',  # prepare to fire on second enemy (1 or 3)',
                3: 'fire',
                4: 'move_up',
                5: 'move_down',
                6: 'move_left',
                7: 'move_right'
}

import random   # random assignement of agent's initial positions
import copy     # make deepcopy of board
import math     # compute distance between agents
from collections import namedtuple

from . import agents

DEBUG = True # set to True for verbose output

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
                flattened_board.append(str(board[(i,j)]))
    return flattened_board

def unflatten(flat_board, agents):
    """ 'Unflatten' the 'flat_board' list to create 'board' dictionary.
    Conventionally, suffix of agent in list represent idx of agent in agent list.
    :params:    flat_board: list of length 121 with -1 for empty case
                agents:     list of (4) agent instances
    :returns:   board:      board dictionary
    """
    board = {}
    n = int(math.sqrt(len(flat_board))) # size of board
    for i in range(n):
        for j in range(n):
            item = flat_board[n*i+j]
            if item == -1:
                board[(i, j)] = None
            else:
                idx = int(item[-1])
                board[(i, j)] = agents[idx]
    return board

def distance(agent1, agent2):
    x1, y1 = agent1.pos
    x2, y2 = agent2.pos
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)             


# Observation: what does each agent know/see?
Observation = namedtuple('Observation', [
    'board',        # flattened board - 121 Ints
    'position',     # position of agent in grid - 2 Ints in [0, 10]
    'alive',        # is agent alive? - Int in [0,1]
    'ammo',         # current ammo level
    'team_mate',    # which agent is teammate - 1 Int in [0,3]
    'enemies'       # which agents are this agent's enemies - 2 Ints in [0, 3]
])

# State: complete state information
State = namedtuple('State', [
    'board',        # flattened board - 121 Ints
    'positions',    # tuple of position tuples for all 4 agents
    'alive',        # tuple of alive flags for all 4 agents
    'ammo',         # tuple of ammo level for all 4 agents
    'aim',          # tuple of aiming for all 4 agents 
])

def string_to_state(str_state):
    """To use a state as key to a dict, it must be
    hashable. The easiest way to do this is to convert
    it to a string:
        str_state = str(state)
    This functions reverses that operation. This is only
    possible because of the highly structured form of
    this string.
    :params:    str_state: a string created from a state
    :returns:   state: an instance of State
    """
    board_string = str_state[13:503]
    # pass: put this on hold until deemed necessary
    # TODO: complete this function if needed

class Environment:
    """The base game environment."""
    def __init__(self, **kwargs):
        self.agents = []
        self.args = kwargs
        self.state = None
        self.action_space = all_actions.copy()
        self.n_actions = len(self.action_space)

        self.set_init_game_state()
    
    def add_agent(self, id_, agent):
        assert isinstance(agent, agents.BaseAgent)
        self.agents.append(agent)
    
    def set_init_game_state(self):
        """Set the initial game state.
        A game state consists of a list of agents states.
        Creates the internal state of the environment.

        First, add all agents

        :returns: a list of observations for the 4 agents
        """
        self.board_size = self.args.get('size', 11) # board size fixed on 11x11
        self.board = {}
        for i in range(self.board_size):
           for j in range(self.board_size):
               self.board[(i,j)] = None
        
        # Position agents randomly on the board
        #for agent in self.agents:
        #    i, j = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
        #    while self.board[(i,j)] is not None:
        #        i, j = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
        #    agent.pos = (i, j)
        #    self.board[(i,j)] = agent

        # Alternative: position agents on fixed initial positions
        positions = [(4, 0), (6, 0), (4, 10), (6, 10)]
        for agent, pos in zip(self.agents, positions):
            agent.pos = pos
            self.board[pos] = agent
        
        # generate the observations for the 4 players
        # by default: player 0 & 1 are 1 team, players 2 & 3 are the other team
        obs = []
        for agent in self.agents:
            observation = self._generate_obs(agent)
            obs.append(observation)
        return obs
    
    def _generate_obs(self, agent):
        if agent.idx in [0, 1]:
            team_mate = 1 if agent.idx == 0 else 0
            enemies = [2, 3]
        else:
            team_mate = 3 if agent.idx == 2 else 2
            enemies = [0, 1]

        observation = Observation(
            board = flatten(self.board),
            position = agent.pos,
            alive = agent.alive,
            ammo = agent.ammo,
            team_mate = team_mate,
            enemies = enemies
        )
        return observation


    def render(self, board=None):
        """Represent the state of the environment. If board is None, use the current
        game state"""
        if board is None:
            board =self.board
        board_repr = ''
        if type(board) == dict:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[(i,j)] == None:
                        board_repr += '  .  '
                    else:
                        board_repr += ' ' + repr(board[(i,j)]) + '  '
                board_repr += '\n'
        elif type(board) == list:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i*self.board_size+j] == -1:
                        board_repr += '  .  '
                    else:
                        board_repr += ' ' + repr(board[i*self.board_size+j]) + '  '
                board_repr += '\n'
        elif type(board) == str:
            board = board[1:-1].split(',')
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[self.board_size*i+j] == ' -1':
                        board_repr += '  .  '
                    else:
                        board_repr += [self.board_size*i+j]
                board_repr += '\n'    
        
        print(board_repr)

    def act(self, observations):
        """Return the chosen action for each agent,
        based on the global observation.
        :params:
            observations: tuple of individual observations
        :returns:
            list of actions, one for each agent in agent_list
        """
        actions = []
        for obs, agent in zip(observations, self.agents):
            actions.append(agent.get_action(obs))
        return actions

    def check_conditions(self, agent, action):
        """Checks whether 'agent' is allowed to execute 'action'"""

        if action == 0 or action == all_actions[0]: # do_nothing
            return True                 # this action is always allowed
        elif action == 1 or action == all_actions[1]: # aim1
            return agent.alive == 1     # only allowed if agent is alive
        elif action == 2 or action == all_actions[2]: # aim1
            return agent.alive == 1     # only allowed if agent is alive
        elif action == 3 or action == all_actions[3]: # fire TODO: check if line-of-sight is free
            if agent.alive == 1 and agent.aim is not None and agent.ammo > 0:
                return True
        elif action == 4 or action == all_actions[4]: # move_up
            if agent.alive == 1 and agent.pos[0] > 0: # stay on the board
                if self.board[(agent.pos[0]-1, agent.pos[1])] is None: # TODO: check case above
                    return True
        elif action == 5 or action == all_actions[5]: # move_down
            if agent.alive == 1 and agent.pos[0] < self.board_size-1: # stay on the board
                if self.board[(agent.pos[0]+1, agent.pos[1])] is None: # TODO: check case below
                    return True
        elif action == 6 or action == all_actions[6]: # move_left
            if agent.alive == 1 and agent.pos[1] > 0: # stay on the board
                if self.board[(agent.pos[0], agent.pos[1]-1)] is None: # TODO: check case left
                    return True
        elif action == 7 or action == all_actions[7]: # move_right
            if agent.alive == 1 and agent.pos[1] < self.board_size-1: # stay on the board
                if self.board[(agent.pos[0], agent.pos[1]+1)] is None: # TODO: check case right
                    return True
        return False # default


    def step(self, actions):
        """Perform actions, part of joint action space.
        Deconflict simultanuous execution of actions (...)
        """
        #board_copy = copy.deepcopy(self.board) # use copy of board to deconflict
        for agent, action in zip(self.agents, actions):
            if not self.check_conditions(agent, action): # if conditions not met => move on to next agent
                if DEBUG:
                    print('action {} not allowed for agent {}'.format(all_actions[action], str(agent)))
                continue
            if action == 0 or action == all_actions[0]: # do_nothing
                pass
            elif action == 1 or action == all_actions[1]: # aim1
                if agent.idx in [0, 1]:
                    agent.aim = 2
                elif agent.idx in [2, 3]:
                    agent.aim = 0
            elif action == 2 or action == all_actions[2]: # aim2
                if agent.idx in [0, 1]:
                    agent.aim = 3
                elif agent.idx in [2, 3]:
                    agent.aim = 1
            elif action == 3 or action == all_actions[3]: # fire
                opponent = self.agents[agent.aim]
                if distance(agent, opponent) < agent.max_range:
                    opponent.alive = 0
                    if DEBUG:
                        print('Agent {} was just killed by {}'.format(
                            str(opponent), str(agent)
                        ))
                agent.ammo -= 1
                agent.aim = None
            elif action == 4 or action == all_actions[4]: # move_up
                self.board[agent.pos] = None
                agent.pos = (agent.pos[0]-1, agent.pos[1])
                self.board[agent.pos] = agent
            elif action == 5 or action == all_actions[5]: # move_down
                self.board[agent.pos] = None
                agent.pos = (agent.pos[0]+1, agent.pos[1])
                self.board[agent.pos] = agent
            elif action == 6 or action == all_actions[6]: # move_left
                self.board[agent.pos] = None
                agent.pos = (agent.pos[0], agent.pos[1]-1)
                self.board[agent.pos] = agent
            elif action == 7 or action == all_actions[7]: # move_right
                self.board[agent.pos] = None
                agent.pos = (agent.pos[0], agent.pos[1]+1)
                self.board[agent.pos] = agent
        
        # generate the observations for the 4 players
        obs = []
        for agent in self.agents:
            observation = self._generate_obs(agent)
            obs.append(observation)
        return obs
    
    def sim_step(self, state, actions):
        """Simulate performing 'actions' on env in state 'state'."""
        self.set_state(state)
        _ = self.step(actions)
        return self.get_state()
    
    def terminal(self):
        """Check if game is over, i.e. when both players of same team are death.
        Conventional:   if team 1 wins, returns  1
                        if team 2 wins, returns -1
                        otherwise, return 0.
        No ties."""
        # if both players of team 1 are death, return -1
        if all(agent.alive == 0 for agent in self.agents[:2]):
            return -1
        # if both players of team 2 are death, return 1
        if all(agent.alive == 0 for agent in self.agents[:2]):
            return 1
        else:
            return 0
    
    def terminal_state(self, state):
        """Check if state is terminal state, by checking alive-status of agents."""
        # TODO: make similar to .terminal(self) (or vice versa)
        # TODO: add tie (return 0) if all agents out of ammo
        if state.alive[0] == 0 and state.alive[1] == 0: # both agents of team 1 are dead
            return -1
        elif state.alive[2] == 0 and state.alive[3] == 0: # both agents of team 2 are dead
            return 1
        else:
            return 0
    
    def get_state(self):
        """Returns complete state information"""
        state = State(
            board = flatten(self.board),
            positions = list(agent.pos for agent in self.agents),
            alive = list(agent.alive for agent in self.agents),
            ammo = list(agent.ammo for agent in self.agents),
            aim = list(agent.aim for agent in self.agents),
        )
        return state
    
    def set_state(self, state):
        """Explicitely set the state of the environment (and agents)"""
        self.board = unflatten(state.board, self.agents)
        for idx, agent in enumerate(self.agents):
            agent.pos = state.positions[idx]
            agent.alive = state.alive[idx]
            agent.ammo = state.ammo[idx]
            agent.aim = state.aim[idx]
        

