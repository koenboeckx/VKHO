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
from collections import defaultdict

named_actions = {0: 'move_up',
                 1: 'move_down',
                 2: 'move_left',
                 3: 'move_right',
}

ACTION_SIZE = len(named_actions)

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
    id = 0
    def __init__(self, depth=2):
        self.id = Hunter.id
        Hunter.id += 1
        self.depth = depth
        self.Q = defaultdict(lambda: [0,]*ACTION_SIZE) # store as Q[obs] = [Q1, ..., QN] for N actions
    
    def get_action(self, obs, T):
        probs = boltzmann(self.Q[obs], T)
        return np.random.choice(range(4), 1, probs)[0]
    
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
        for agent in self.hunters + self.prey:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            agent.x, agent.y = x, y
    
    def get_init_state(self):
        return self.get_board(self.hunters + self.prey)
    
    def render(self, board):
        s = ''
        for x in range(self.size):
            for y in range(self.size):
                if board[(x,y)] is None:
                    s += ' .. '
                else:
                    s += ' ' + str(board[(x,y)]) + ' '
            s += '\n'
        print(s)
    
    def get_board(self, agents):
        board = {(i,j): None for i in range(self.size) for j in range(self.size)}
        for agent in agents:
            x, y = agent.x, agent.y
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
        for agent, action in zip(self.hunters+self.prey, h_actions+p_actions):
            if action == 0: # move up
                agent.y = max(min(agent.y-1, self.size-1), 0)
            elif action == 1: # move down
                agent.y = max(min(agent.y+1, self.size-1), 0)
            elif action == 2: # move left
                agent.x = max(min(agent.x-1, self.size-1), 0)
            elif action == 3: # move left
                agent.x = max(min(agent.x+1, self.size-1), 0)
            else:
                raise ValueError('action {} not allowed'.format(action))
        
        # for each hunter, get window and derive sensation
        sensations = {}
        for hunter in self.hunters:
            sensations[hunter] = None
            window = self.get_window(hunter)
            for x, y in window:
                for p in self.prey:
                    if x == p.x and y == p.y:
                        sensations[hunter] = p # this way: only last prey found is returned

        # reconstruct board
        board = self.get_board(self.hunters + self.prey)

        return board, sensations
    
    def get_window(self, agent):
        x, y = agent.x, agent.y
        depth = agent.depth
        window = []
        for i in range(-depth, depth+1):
            if 0<= x + i < self.size:
                for j in range(-depth, depth+1):
                    if 0<= y + j < self.size:
                        window.append((x+i, y+j))
        return window
    
    def terminal(self):
        for hunter in self.hunters:
            for p in self.prey:
                if hunter.x == p.x and hunter.y == p.y:
                    return True
        return False              

if __name__ == '__main__':
    hunters = [Hunter() for _ in range(2)]
    prey    = [Prey() for _ in range(2)]
    env = Environment(hunters, prey)
    state = env.get_init_state()
    
    while not env.terminal():
        h_actions = [h.get_action(0, 1) for h in hunters]
        p_actions = [p.get_action(0) for p in prey]
        state, obs = env.step(state, h_actions, p_actions)
        env.render(state)
        print(obs)
        
