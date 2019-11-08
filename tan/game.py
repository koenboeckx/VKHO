"""
All the tasks considered in this study involve
hunter agents seeking to capture randomly-
moving prey agents in a 10 by 10 by 10 grid world.
On each time step, each agent (hunter or prey) has
four possible actions to choose from: moving up, 
down, right or left within the boundary. More than
one agent can occupy the same cell. A prey is
captured when:
(1) it occupies the same cell as a hunter, or
(2) when two hunters occupy the same cell as the prey
or are next to the prey.
Upon capturing the prey, the hunter(s) receive +1 reward.
They receive -0.1 reward for each move when they do not
capture a prey.
Each hunter has a limited visual field inside which
it can locate prey accurately. Each hunter's sensation
is represented by (x,y) where x (y) is the relative distance
of the closest prey along the x (y) axis. If two prey 
are equally close to a hunter, only one of them (chosen 
randomly) will be sensed. If there is not prrey
in sight, a unique default sensation is used.
"""

import random
import math
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

named_actions = {0: 'move_up',
                 1: 'move_down',
                 2: 'move_left',
                 3: 'move_right',
}

ACTION_SIZE = len(named_actions)

N_HUNTERS = 2
N_PREY = 2

NORM = 2 # normalize circles (esthetics only)

def visualize(agent, temp=0.4, title=None):
    """
    Shows how the different Q-values steer the behavior
    of the agent in the different potential states.
    """
    positions = [(i, j)  for i in range(-agent.depth, agent.depth+1)
                         for j in range(-agent.depth, agent.depth+1)]
    centers = [(.5, .75), (.5, .25), (.25, .5), (.75, .5)] # top, down left, right

    _, axarr = plt.subplots(2*agent.depth+1, 2*agent.depth+1)
    for i, j in positions:
        action_values = agent.Q[(i,j)]
        probs = boltzmann(action_values, temp)
        for idx, p in enumerate(probs):
            circle = plt.Circle( centers[idx], p / NORM )
            axarr[agent.depth + j, agent.depth + i].add_patch(circle)
        axarr[agent.depth + j, agent.depth + i].set_title("{}, {}".format(j, i))
        axarr[agent.depth + j, agent.depth + i].axis('off')
    
    if title is not None:
        plt.title(title)
        
    plt.show()

def boltzmann(Qs, T):
    probs = [math.exp(q/T) for q in Qs]
    prob_sum = sum(probs)
    return [p/prob_sum for p in probs]

class Agent:
    def __init__(self):
        pass

    def get_action(self, obs):
        raise NotImplementedError

class Hunter(Agent):
    """
    Container for:  (1) get_action method
                    (2) Q-dict to store Q(s,a) value
    """
    id = 0
    def __init__(self, depth=2):
        self.id = Hunter.id
        Hunter.id += 1
        self.depth = depth
        self.Q = defaultdict(lambda: [0,]*ACTION_SIZE) # store as Q[obs] = [Q1, ..., QN] for N actions
    
    def get_action(self, obs, T):
        probs = boltzmann(self.Q[obs], T)
        return np.random.choice(range(4), 1, p=probs)[0]
    
    def __repr__(self):
        return 'H'+str(self.id)
    
    __str__ = __repr__

class Prey(Agent):
    id = 0
    def __init__(self):
        self.id = Prey.id
        Prey.id += 1
    
    def __repr__(self):
        return 'P'+str(self.id)
    
    def get_action(self, obs):
        return random.randint(0, 3)
    
    __str__ = __repr__

class Environment:
    def __init__(self, hunters, prey, **kwargs):
        self.hunters = hunters
        self.prey = prey

        self.size = kwargs.get('size', 10)
    
    def get_init_state(self):
        state = {}
        for agent in self.hunters + self.prey:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            state[agent] = (x, y)
        sensations = self.get_sensations(state)
        #board = self.get_board(state)
        return state, sensations
    
    def render(self, state):
        board = self.get_board(state)
        s = ''
        for x in range(self.size):
            for y in range(self.size):
                if board[(x,y)] is None:
                    s += ' .. '
                else:
                    s += ' ' + str(board[(x,y)]) + ' '
            s += '\n'
        print(s)
    
    def get_board(self, state):
        board = {(i,j): None for i in range(self.size) for j in range(self.size)}
        for agent in self.hunters + self.prey:
            x, y = state[agent]
            board[(x, y)] = agent
        return board

    def step(self, state, h_actions, p_actions):
        """
        Execute one step in thhe environment.

        :param state: the current state in which the actions are to be performed
        :param h_actions: tuple of actions to be executed by the hunters
        :param p_actions: tuple of actions to be executed by the prey

        :return: tuple (state, observations) where observations is a tuple of
                    sensations for the different hunters
        """
        new_state = {}
        for agent, action in zip(self.hunters+self.prey, h_actions+p_actions):
            x, y = state[agent]
            if action == -1: # don't move / for testing
                new_x, new_y = x, y 
            elif action == 0: # move up
                new_x, new_y = x, max(min(y-1, self.size-1), 0)
            elif action == 1: # move down
                new_x, new_y = x, max(min(y+1, self.size-1), 0)
            elif action == 2: # move left
                new_x, new_y = max(min(x-1, self.size-1), 0), y
            elif action == 3: # move right
                new_x, new_y = max(min(x+1, self.size-1), 0), y
            else:
                raise ValueError('action {} not allowed'.format(action))
        
            new_state[agent] = (new_x, new_y)
        
        sensations = self.get_sensations(new_state)
        
        return new_state, sensations
    
    def get_window(self, state, agent):
        depth = agent.depth
        x, y = state[agent]
        window = []
        for i in range(-depth, depth+1):
            if 0<= x + i < self.size:
                for j in range(-depth, depth+1):
                    if 0<= y + j < self.size:
                        window.append((i, j)) # TODO: check this - not .append((x+i, y+j))?
        return window
    
    def get_sensations(self, state):
        # for each hunter, get window and derive sensation
        sensations = {}
        for hunter in self.hunters:
            sensations[hunter] = None
            window = self.get_window(state, hunter)
            for i, j in window:
                for p in self.prey:
                    if state[hunter][0] + i == state[p][0] and  state[hunter][1] + j == state[p][1]:
                        sensations[hunter] = (i, j) # this way: only last prey found is returned
        return sensations
    
    def captured(self, state):
        """
        A prey is captured when it occupies the same cell as a hunter
        or when two hunters either occupy the same cell as the prey 
        or are next to the prey.
        """
        # 1. check if hunter and prey on same cell
        for prey in self.prey:
            xp, yp = state[prey]
            for hunter in self.hunters:
                xh, yh = state[hunter]
                if xp == xh and yp == yh:
                    return True
        
        # 2. Check if two hunters are next to prey
        for prey in self.prey:
            xp, yp = state[prey]
            for hunter1 in self.hunters:
                xh1, yh1 = state[hunter1]
                if xh1 in [xp-1, xp+1] and yh1 in [yp-1, yp+1]:
                    for hunter2 in [h for h in self.hunters if h is not hunter1]:
                        xh2, yh2 = state[hunter2]
                        if xh2 in [xp-1, xp+1] and yh2 in [yp-1, yp+1]:
                            return True
        return False
    
    def get_reward(self, state):
        "If a prey is captured, return 1. Else return -0.1"
        return 1 if self.captured(state) else -0.1
    
    def terminal(self, state):
        "Game is over when a prey is captured."
        return self.captured(state)
               

def create_game(n_hunters, n_prey, depth=2):
    hunters = [Hunter(depth=depth) for _ in range(n_hunters)]
    prey    = [Prey() for _ in range(n_prey)]
    env = Environment(hunters, prey)
    return hunters, prey, env

if __name__ == '__main__':
    hunters, prey, env = create_game(N_HUNTERS, N_PREY)
    #visualize(hunters[0])

    """
    state, _ = env.get_init_state()
    
    while not env.terminal(state):
        h_actions = [h.get_action(0, 1) for h in hunters]
        p_actions = [p.get_action(0) for p in prey]
        state, obs = env.step(state, h_actions, p_actions)
        env.render(state)
        print(obs)
    
    print(boltzmann([.1, .0, .0, .0], 0.4))
    print(boltzmann([.1, .0, .0, .0], 0.01))
    """

    probs = [0.0015024974129203073, 0.0007072329756351492, 0.996921668937134, 0.0008686006743105232]
    for _ in range(100):
        print(np.random.choice(range(4), 1, p=probs)[0])
