"""
v0 of the game Environment.
Limitations:
    * Fully observable, no terrain
    * Fixed size board (11x11)
    * Fixed teams: 2 against 2

15/10/19:   reduced version of environment => act(state, actions) -> new_state
            no more 'Agent' objects
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

DEBUG_ENV = False # set to True for verbose output
MAX_RANGE = 8 # maximum range of Agents weapon => big impact on termination speed

# State: complete state information, including next player
State = namedtuple('State', [
    'board',        # flattened board - 121 Ints
    'positions',    # tuple of position tuples for all 4 agents
    'alive',        # tuple of alive flags for all 4 agents
    'ammo',         # tuple of ammo level for all 4 agents
    'aim',          # tuple of aiming for all 4 agents 
    'player',       # player to move in this state
])

# helper functions
def print_state(state):
    print('Positions = ', state.positions)
    print('Aims = ', state.aim)
    print('Ammo = ', state.ammo)
    print('Alive = ', state.alive)
    print('Player = ', state.player)

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
    return tuple(flattened_board)

def unflatten(state):
    """ 'Unflatten' the 'flat_board' list in state.board to create a 'board' dictionary.
    Conventionally, suffix of agent in list represent idx of agent in agent list.
    :params:    flat_board: list of length 121 with -1 for empty case
                agents:     list of (4) agent instances
    :returns:   board:      board dictionary
    """
    flat_board = state.board
    board = {}
    n = int(math.sqrt(len(flat_board))) # size of board
    for i in range(n):
        for j in range(n):
            item = flat_board[n*i+j]
            if item == -1:
                board[(i, j)] = None
            else:
                idx = int(item[-1])
                board[(i, j)] = idx
    return board

def distance(state, agent1, agent2):
    pos1 = state.positions[agent1]
    pos2 = state.positions[agent2]
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)             

class Environment:
    """The base game environment."""
    def __init__(self, **kwargs):
        self.agents = []
        self.args = kwargs
        self.state = None
        self.action_space = all_actions.copy()
        self.n_actions = len(self.action_space)
    
    def get_init_game_state(self, random_pos=False):
        """Get the initial game state.
        A game state consists of a list of agents states.

        First, add all agents

        :returns: a list of observations for the 4 agents
        """
        board_size = self.args.get('size', 11) # board size fixed on 11x11
        agents = (0, 1, 2, 3)
        board = {}
        for i in range(board_size):
           for j in range(board_size):
               board[(i,j)] = None
        
        # Position agents randomly on the board
        if random_pos:
            positions = []
            for agent in agents:
                i, j = random.randint(0, board_size-1), random.randint(0, board_size-1)
                while board[(i,j)] is not None:
                    i, j = random.randint(0, board_size-1), random.randint(0, board_size-1)
                positions.append((i, j))
                board[(i,j)] = agent
        else:
            # Alternative: position agents on fixed initial positions
            positions = [(4, 0), (6, 0), (4, 10), (6, 10)]
            for agent, pos in zip(agents, positions):
                board[pos] = agent
        
        # generate the state 
        # by default, player 0 begins to play
        state = State(
            board = flatten(board),
            positions = tuple(positions),
            alive = (1, 1, 1, 1), 
            ammo = (5000, 5000, 5000, 5000),
            aim = (None, None, None, None),
            player = 0,
        )
        return state

    def render(self, state):
        """Represent the state of the environment."""
        board = state.board
        board_size = self.args.get('size', 11)
        board_repr = ''
        if type(board) in [list, tuple]:
            for i in range(board_size):
                for j in range(board_size):
                    if board[i*board_size+j] == -1:
                        board_repr += '  .  '
                    else:
                        board_repr += ' ' + repr(board[i*board_size+j]) + '  '
                board_repr += '\n'
        elif type(board) == str:
            board = board[1:-1].split(',')
            for i in range(board_size):
                for j in range(board_size):
                    if board[board_size*i+j] == ' -1':
                        board_repr += '  .  '
                    else:
                        board_repr += [board_size*i+j]
                board_repr += '\n'    
        
        print(board_repr)

    def check_conditions(self, state, agent, action):
        """Checks whether 'agent' is allowed to execute 'action' in with game in 'state'"""
        board = unflatten(state)
        board_size = self.args.get('size', 11)

        if action == 0 or action == all_actions[0]: # do_nothing
            return True                 # this action is always allowed
        elif action == 1 or action == all_actions[1]: # aim1
            return state.alive[agent] == 1     # only allowed if agent is alive
        elif action == 2 or action == all_actions[2]: # aim1
            return state.alive[agent] == 1     # only allowed if agent is alive
        elif action == 3 or action == all_actions[3]: # fire TODO: check if line-of-sight is free
            if state.alive[agent] == 1 and state.aim[agent] is not None and state.ammo[agent] > 0:
                return True
        elif action == 4 or action == all_actions[4]: # move_up
            if state.alive[agent] == 1 and state.positions[agent][0] > 0: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0]-1, agent_pos[1])] is None: # TODO: check case above
                    return True
        elif action == 5 or action == all_actions[5]: # move_down
            if state.alive[agent] == 1 and state.positions[agent][0] < board_size-1: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0]+1, agent_pos[1])] is None: # TODO: check case below
                    return True
        elif action == 6 or action == all_actions[6]: # move_left
            if state.alive[agent] == 1 and state.positions[agent][1] > 0: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0], agent_pos[1]-1)] is None: # TODO: check case left
                    return True
        elif action == 7 or action == all_actions[7]: # move_right
            if state.alive[agent] == 1 and state.positions[agent][1] < board_size-1: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0], agent_pos[1]+1)] is None: # TODO: check case right
                    return True
        return False # default


    def step(self, state, actions):
        """Perform actions, part of joint action space.
        Deconflict simultanuous execution of actions (...)
        """
        board = unflatten(state)
        board_size = self.args.get('size', 11)
        agents = (0, 1, 2, 3)

        aim = list(state.aim)
        alive = list(state.alive)
        ammo = list(state.ammo)
        positions = list(state.positions)

        for agent, action in zip(agents, actions):
            if not self.check_conditions(state, agent, action): # if conditions not met => move on to next agent
                if DEBUG_ENV:
                    print('action {} not allowed for agent {}'.format(all_actions[action], str(agent)))
                continue
            if action == 0 or action == all_actions[0]: # do_nothing
                pass
            elif action == 1 or action == all_actions[1]: # aim1
                if agent in [0, 1]:
                    aim[agent] = 2
                elif agent in [2, 3]:
                    aim[agent] = 0
            elif action == 2 or action == all_actions[2]: # aim2
                if agent in [0, 1]:
                    aim[agent] = 3
                elif agent in [2, 3]:
                    aim[agent] = 1
            elif action == 3 or action == all_actions[3]: # fire
                opponent = state.aim[agent]
                if distance(state, agent, opponent) < MAX_RANGE: # simple kill criterion
                    alive[opponent] = 0
                    if DEBUG_ENV:
                        print('Agent {} was just killed by {}'.format(
                            str(opponent), str(agent)
                        ))
                ammo[agent] -= 1
                aim[agent] = None
            elif action == 4 or action == all_actions[4]: # move_up
                agent_pos = state.positions[agent]
                board[state.positions[agent]] = None
                positions[agent] = (agent_pos[0]-1, agent_pos[1])
                board[positions[agent]] = agent
            elif action == 5 or action == all_actions[5]: # move_down
                agent_pos = state.positions[agent]
                board[state.positions[agent]] = None
                positions[agent] = (agent_pos[0]+1, agent_pos[1])
                board[positions[agent]] = agent
            elif action == 6 or action == all_actions[6]: # move_left
                agent_pos = state.positions[agent]
                board[state.positions[agent]] = None
                positions[agent] = (agent_pos[0], agent_pos[1]-1)
                board[positions[agent]] = agent
            elif action == 7 or action == all_actions[7]: # move_right
                agent_pos = state.positions[agent]
                board[state.positions[agent]] = None
                positions[agent] = (agent_pos[0], agent_pos[1]+1)
                board[positions[agent]] = agent
        
        board = flatten(board)
        new_state = State(board=board,
                        positions=tuple(positions),
                        alive=tuple(alive),
                        ammo=tuple(ammo),
                        aim=tuple(aim),
                        player=1-state.player) # switch player
        return new_state

    def terminal(self, state):
        """Check if state is terminal state, by checking alive-status of agents."""
        # TODO: make similar to .terminal(self) (or vice versa)
        # TODO: add tie (return 0) if all agents out of ammo
        if state.alive[0] == 0 and state.alive[1] == 0: # both agents of team 1 are dead
            return -1
        elif state.alive[2] == 0 and state.alive[3] == 0: # both agents of team 2 are dead
            return 1
        else:
            return 0
