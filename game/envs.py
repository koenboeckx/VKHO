"""
v0 of the game Environment.
Limitations:
    * Fully observable, no terrain
    * Fixed size board (default = 11x11)
    * Fixed teams: 2 against 2

15/10/19:   reduced version of environment => act(state, actions) -> new_state
            no more 'Agent' objects
"""

DEBUG_ENV = False
DEBUG_ENV_2 = False

all_actions = { 0: 'do_nothing',
                1: 'aim0',  # prepare to fire on first  enemy (0 or 2)
                2: 'aim1',  # prepare to fire on second enemy (1 or 3)',
                3: 'fire',
                4: 'move_north',
                5: 'move_south',
                6: 'move_west',
                7: 'move_east'
}

import random   # random assignement of agent's initial positions
import copy     # make deepcopy of board
import math     # compute distance between agents
import numpy as np  # used in computing LOS
from collections import namedtuple
import pickle

"""
try:
    from . import agents
except:
    print('Running from file')
    import agents
"""


import agents # temporary, to avoid error from above when running in debug mode
# helper functions

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
class State:
    def __init__(self, board, positions, alive, ammo, aim):
        self.board = board
        self.positions = positions
        self.alive = alive 
        self.ammo = ammo
        self.aim = aim
    
    def __str__(self):
        s = "Positions = {}\nAims = {}\nAmmo = {}\nAlive = {}".format(
            self.positions, self.aim, self.ammo, self.alive
        )
        return s


def flatten(board):
    """Flatten the 'board' dictionary
    :params:    board: dictionary containing different elements
    :return:    flattened_board: 121 Ints
    """
    n = int(math.sqrt(len(board)))
    flattened_board = []
    for i in range(n):
        for j in range(n):
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
    """
    Compute euclidean distance between two agents
    :param state: instance of Stace
    :param agent1: index of first agent
    :param agent2: index of second agent

    :return: distance between agents
    """
    pos1 = state.positions[agent1]
    pos2 = state.positions[agent2]
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2) 


def get_line_of_sight_dict(size, remove_endpoints=True):
    """
    Pre-compute the straight lines between all two points.

    :param size: size of the board
    :param remove_endpoints: if True, remove the endpoints from LOS
    :return: a dict with keys = ((xi, yi), (xj, yj)) pairs
             and values = [..., (xk, yk), ...] element along
             line between (xi, yi) and (xj, yj).
    """
    try: # if pickle file already present, open this => speed up
        los = pickle.load(open(f"game/los/los_{size}.p", "rb" ) )
    except:
        N = 20 # number of points considered, 100 = randomly chosen value
        los = {}
        all_squares = [(x, y) for x in range(size) for y in range(size)]
        for (xi, yi) in all_squares:
            for (xj, yj) in all_squares:
                los[((xi, yi), (xj, yj))] = []
                dx = (xj - xi)/N
                dy = (yj - yi)/N
                xs = [xi+k*dx for k in range(N)]
                ys = []
                for k, x in zip(range(N), xs): # create list of consecutive, non-integer x values
                    if x == xi: # move in horizontal line
                        ys.append(yi + k*dy)
                    else:
                        ys.append(yi + (yj - yi) * (x - xi) /(xj - xi))
                for x, y in zip(xs, ys):
                    los[((xi, yi), (xj, yj))].append((int(round(x)), int(round(y))))
                """
                for x_test, y_test in all_squares:
                    for x, y in zip(xs, ys):
                        if  x_test-.5 <= x <= x_test+.5 and y_test-.5 <= y <=  y_test+.5:
                            los[((xi, yi), (xj, yj))].append((x_test, y_test))
                """
                los[((xi, yi), (xj, yj))] = list(set(los[((xi, yi), (xj, yj))])) # remove doubles
                # remove endpoints:
                if remove_endpoints:
                    if (xi, yi) in los[((xi, yi), (xj, yj))]:
                        los[((xi, yi), (xj, yj))].remove((xi, yi))
                    if (xj, yj) in los[((xi, yi), (xj, yj))]:
                        los[((xi, yi), (xj, yj))].remove((xj, yj))

        pickle.dump( los, open( f"game/los/los_{size}.p", "wb+" ) )
    return los


class Environment:
    """The base game environment."""
    def __init__(self, agents, **kwargs):
        self.agents = agents
        self.args = kwargs
        self.max_range = kwargs.get('max_range', 8)
        self.state = None
        self.action_space = all_actions.copy()
        self.n_actions = len(self.action_space)
        self.board_size = self.args.get('size', 11) # board size fixed on 11x11
        self.los = get_line_of_sight_dict(self.board_size)
        self.step_penality = kwargs.get('step_penality', 0.0) # to induce shorter episodes

        for agent in agents:
            agent.env = self # ad reference to environment in each agent
    
    def get_init_game_state(self):
        """Get the initial game state.
        :returns: a list of observations for the 4 agents
        """

        # reintialize agents
        for agent in self.agents:
            agent.init_agent(agent.idx)
        
        agents = (0, 1, 2, 3)
        board = {}
        for i in range(self.board_size):
           for j in range(self.board_size):
               board[(i,j)] = None
        
        # Alternative: position agents on fixed initial positions
        high = self.board_size//2 - 1
        low  = self.board_size//2 + 1
        positions = [(high, 0), (low, 0),
                    (high, self.board_size-1), (low, self.board_size-1)]
        for agent, pos in zip(agents, positions):
            board[pos] = agent
        
        # generate the state 
        # by default, player 0 begins to play
        state = State(
            board = flatten(board),
            positions = tuple(positions),
            alive = tuple(agent.alive for agent in self.agents), 
            ammo = tuple(agent.ammo for agent in self.agents),
            aim = tuple(agent.aim for agent in self.agents),
        )
        return state

    def render(self, state):
        """Represent the state of the environment.
        :param state: the state to show (actually only state.board)
        """
        board = state.board
        board_size = self.args.get('size', 11)
        board_repr = ''
        if type(board) in [list, tuple]:
            for i in range(board_size):
                for j in range(board_size):
                    if board[i*board_size+j] == -1:
                        board_repr += ' . '
                    else:
                        board_repr += ' ' + board[i*board_size+j] + ' '
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
        board_repr += f"alive = {state.alive} - ammo = {state.ammo} - aims = {state.aim}"
        print(board_repr)
    
    def get_unavailable_actions(self, state, agent):
        if isinstance(agent, int):
            unavailable = [action for action in all_actions.keys() 
                        if self.check_conditions(state, agent, action) is False]
        else:
            unavailable = [action for action in all_actions.keys() 
                        if self.check_conditions(state, agent.idx, action) is False]
        return unavailable

    def check_conditions(self, state, agent, action):
        """
        Checks whether 'agent' is allowed to execute 'action' in with game in 'state'
        :param state: current state of the game
        :param agent: check the conditions for this agent 
        :param action: check the conditions for the action of agent

        :return: True if action is allowed else False 
        """
        board = unflatten(state)
        board_size = self.args.get('size', 11)

        if action == 0 or action == all_actions[0]: # do_nothing
            return True                 # this action is always allowed
        elif action == 1 or action == all_actions[1]: # aim1
            return state.alive[agent] == 1     # only allowed if agent is alive
        elif action == 2 or action == all_actions[2]: # aim1
            return state.alive[agent] == 1     # only allowed if agent is alive
        elif action == 3 or action == all_actions[3]: # fire
            if state.alive[agent] == 0:
                return False                                                      
            if state.aim[agent] is None:
                return False
            if state.ammo[agent] <= 0:
                return False

            opponent = [0, 1, 2, 3][state.aim[agent]]
            x, y = state.positions[agent]
            x_opp, y_opp = state.positions[opponent]
            los = self.los[((x, y), (x_opp, y_opp))]
            if self.check_window(state, los): # check if LOS is clear
                return True
            return False # default response for fire unless conditions above are met
        elif action == 4 or action == all_actions[4]: # move_up
            if state.alive[agent] == 1 and state.positions[agent][0] > 0: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0]-1, agent_pos[1])] is None: 
                    window = self.get_window(agent_pos[0]-1, agent_pos[1])
                    window.remove((agent_pos[0], agent_pos[1])) # remove future pos (vacant anyway)
                    if self.check_window(state, window):
                        return True
        elif action == 5 or action == all_actions[5]: # move_down
            if state.alive[agent] == 1 and state.positions[agent][0] < board_size-1: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0]+1, agent_pos[1])] is None: 
                    window = self.get_window(agent_pos[0]+1, agent_pos[1])
                    window.remove((agent_pos[0], agent_pos[1])) # remove future pos (vacant anyway)
                    if self.check_window(state, window):
                        return True
        elif action == 6 or action == all_actions[6]: # move_left
            if state.alive[agent] == 1 and state.positions[agent][1] > 0: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0], agent_pos[1]-1)] is None: 
                    window = self.get_window(agent_pos[0], agent_pos[1]-1)
                    window.remove((agent_pos[0], agent_pos[1])) # remove future pos (vacant anyway)
                    if self.check_window(state, window):
                        return True
        elif action == 7 or action == all_actions[7]: # move_right
            if state.alive[agent] == 1 and state.positions[agent][1] < board_size-1: # stay on the board
                agent_pos = state.positions[agent]
                if board[(agent_pos[0], agent_pos[1]+1)] is None: 
                    window = self.get_window(agent_pos[0], agent_pos[1]+1)
                    window.remove((agent_pos[0], agent_pos[1])) # remove future pos (vacant anyway)
                    if self.check_window(state, window):
                        return True
        return False # default

    def check_window(self, state, window):
        """Returns True is winndow is unobstructed in state; False otherwise."""
        for pos in window:
            if pos in state.positions:
                return False
        return True

    def get_window(self, x, y, depth=1):
        """
        Return window  around (x, y) (size = (depth+2)**2-1)
        :param x, y: coordinates around which to compute window
        :param depth: size of the window
        :return: list of positions (x_win, y_win)
        """
        window = []
        for i in range(-depth, depth+1):
            if 0 <= x + i < self.board_size:
                for j in range(-depth, depth+1):
                    if 0 <= y + j < self.board_size:
                        window.append((x+i, y+j))
        window.remove((x, y)) # remove own position
        return window
    
    def get_actions(self, state): # TODO: change this to work with observations
        # return [agent.get_action(state) for agent in self.agents  ] # next line takes alive into account
        return [agent.get_action(state) if state.alive[agent.idx] else 0 for agent in self.agents]

    def get_observation(self, state, agent):
        """
        Returns observation of state for agent.
        :param state:   instance of State
        :param agent:   instance of Agent
        :returns:       nd.array of size (4, 2*agent.obs_space+1, 2*agent.obs_space+1)
                        containing:
                        * plane 0: self + agents of own team (0/1 indicator; -1 if not accesible)
                        * plane 1: (normalized) ammo of own team
                        * plane 2: agents of other team (0/1 indicator; -1 if not accesible)
                        * plane 3: (normalized) ammo of other team
        """
        obs = np.zeros((4, 2*agent.obs_space+1, 2*agent.obs_space+1))
        agent_pos = state.positions[agent.idx]
        view = (agent_pos[0] - agent.obs_space,
                agent_pos[0] + agent.obs_space,
                agent_pos[1] - agent.obs_space,
                agent_pos[1] + agent.obs_space)
        for other in self.agents:
            if state.alive[other.idx] == 0: # skip dead agents
                continue
            other_pos = state.positions[other.idx]
            if view[0] <=  other_pos[0] <= view[1]:
                if view[2] <=  other_pos[1] <= view[3]: # check if other agent is in view
                    x_rel = other_pos[0] - agent_pos[0]
                    y_rel = other_pos[1] - agent_pos[1]
                    if other.team == agent.team:
                        obs[0, agent.obs_space+x_rel, agent.obs_space+y_rel] = 1.
                        obs[1, agent.obs_space+x_rel, agent.obs_space+y_rel] = state.ammo[other.idx]/other.init_ammo
                    else:
                        obs[2, agent.obs_space+x_rel, agent.obs_space+y_rel] = 1.
                        obs[3, agent.obs_space+x_rel, agent.obs_space+y_rel] = state.ammo[other.idx]/other.init_ammo
        # for all positions not accesible (off-board), set value to -1
        for x in range(-agent.obs_space, agent.obs_space+1):
            for y in range(-agent.obs_space, agent.obs_space+1):
                if x + agent_pos[0] < 0 or x + agent_pos[0] >= self.board_size:
                    obs[:, x+agent.obs_space, y+agent.obs_space] = -1.
                if y + agent_pos[1] < 0 or y + agent_pos[1] >= self.board_size:
                    obs[:, x+agent.obs_space, y+agent.obs_space] = -1.
        return obs                    

    def get_all_obs(self, state):
        return [self.get_observation(state, agent) for agent in self.agents]

    def step(self, state, actions):
        """Perform actions, part of joint action space.
        :param state: the game state in which the action is to be executed
        :param actions: tuple of 4 actions that the players execute

        :return: next game state
        """
        board = unflatten(state)
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
                if distance(state, agent, opponent) < self.agents[agent].max_range: # simple kill criterion
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
                        aim=tuple(aim))
        return new_state

    def terminal(self, state):
        """Check if state is terminal state, by checking alive-status of agents.
        :param state: current game state
        
        :return: 1 if player 0 wins, -1 if player 1 wins, 0 otherwise
        """
        # both agents of team 0 are dead or out of ammo
        if all([alive*ammo == 0 for alive, ammo in zip(state.alive[:2], state.ammo[:2])]):
            return -1
        # both agents of team 1 are dead or out of ammo
        elif all([alive*ammo == 0 for alive, ammo in zip(state.alive[2:], state.ammo[2:])]):
            return 1 
        else:
            return 0
    
    def get_reward(self, state):
        """Return tuple of rewards, one for each agent.
        :param: state
        
        :return: list of rewards
        """
        reward = self.terminal(state)
        if reward == 'out-of-ammo':
            return (-1, -1, -1, -1)
        elif reward == 0:
            penality = self.step_penality
            return (-penality, -penality, -penality, -penality)
        else:
            return (reward, reward, -reward, -reward)

if __name__ == '__main__':
    agent_params = {
        'init_ammo':            5,
        'view_size':            7,
        'max_range':            5,
    }
    params = {
        'board_size':           7,
        'step_penalty':         0.01,
    }
    class Tank:
        def __init__(self, idx, team="friend"):
            super(Tank, self).__init__()
            self.init_agent(idx)
            self.team = team
            
        
        def init_agent(self, idx):
            self.type = 'T'
            self.idx  = idx

            # specific parameters
            self.alive = 1
            self.init_ammo = agent_params['init_ammo']
            self.ammo = agent_params['init_ammo']
            self.max_range = agent_params['max_range']
            self.obs_space = agent_params['view_size']
            self.pos = None     # initialized by environment
            self.aim = None     # set by aim action
            
        
        def __repr__(self):
            return self.type + str(self.idx)
        
        def save(self, filename):
            with open(filename, 'wb') as output_file:
                pickle.dump(self, output_file)

    class RandomTank(Tank):
        def __init__(self, idx, team="friend"):
            super(RandomTank, self).__init__(idx, team)
        
        def get_action(self, obs):
            return random.randint(0, 7)
    
    agents = [RandomTank(0, team="friend"), RandomTank(1, team="friend"),
              RandomTank(2, team="enemy"), RandomTank(3, team="enemy")]
    env = Environment(agents, size=params['board_size'],
                        step_penality=params['step_penalty'])
    state = env.get_init_game_state()
    for idx in range(10):
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        all_obs = env.get_all_obs(next_state)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) != 0 else False
        state = next_state