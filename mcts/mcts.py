"""
Attempt to solve - or find non-trivial tactics - for the game
For now: only symmetric game with 2 Tanks per Player (=Commander)
"""

import time
import copy
import math # .sqrt, .log to compute UCB
import random # for generating random moves during rollout
import numpy as np # infinity
from itertools import product   # create joint action spaces

import game
from game.agents import Tank

DEBUG = False # set to True for verbose output

def obs_to_int(obs):
    """convert an observation to a unique integer. Here, an observation
    if the combined observation of the two agents on one team"""
    return hash(str(obs))

def state_to_int(state):
    """convert a state namedtuple to a unique integer."""
    return hash(str(state))

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
        self.max_search_time = 1 # max search time in seconds => proxy for search depth

    def add_agent(self, agent):
        self.agents.append(agent)
    
    def set_environment(self, env):
        self.env = copy.deepcopy(env) # environment used for simulation
        self.action_space = list(product(env.action_space.keys(),
                                         env.action_space.keys()))
    
    def get_actions(self, team, state):
        """Run MCTS from state. After max time expires, return action with 
        highest/lowest q-value depending on team"""
        state_int = state_to_int(state)
        # if state not in self.n_visits, add it
        if state_int not in self.n_visits:
            self.n_visits[state_int] = 0
            self.value[state_int] = 0
        
        # run x iterations of MCTS starting from 'state'
        start_time = time.time()
        #while time.time() - start_time < self.max_search_time:
        for i in range(2): # remove this line later
            self.one_iteration(team, state)
        
        # now select the best action
        if team == 0:
            _, best_action = max([(self.q_value[(state_int, a)], a) for a in self.env.action_space])
        else:
            _, best_action = min([(self.q_value[(state_int, a)], a) for a in self.env.action_space])
        return best_action

    def one_iteration(self, team, start_node):
        current_state = start_node
        while not self.is_leaf(current_state):
            # construct list of successor states
            child_nodes = [self.env.sim_step(current_state, team, actions)
                          for actions in self.action_space]

            # compute UCB and select best action
            best = max if team == 0 else min
            _, best_action = best([(self.ucb(current_state, action), action)
                                for action in self.action_space])
            current_state = self.env.sim(current_state, best_action)
        
        state_int = state_to_int(current_state)

        if self.n_visits[state_int] == 0: # first visit to this node => perform rollout and backprop
            reward = self.rollout(current_state, team)
            self.n_visits[state_int] += 1
            self.value[state_int] += reward

        # from here: next_state is a leaf state
        self.expanded.append(state_int)

    def rollout(self, state, team):
        while self.env.terminal_state(state) == 0: # no team has both players dead
            team = 0 if team == 1 else 1 # switch teams after each round
            # generate random action
            action = (random.randint(0, 7), random.randint(0, 7)) # TODO: remove hardcoded 7 
            state = self.env.sim_step(state, team, action)
            if DEBUG:
                self.env.render(state.board)
                print('alive = ', state.alive)
                print('ammo  = ', state.ammo)
        return self.env.terminal_state(state)
    
    def is_leaf(self, state):
        return state_to_int(state) not in self.expanded 

    def ucb(self, state, action):
        state_int = state_to_int(state)
        if state_int not in self.n_visits:
            return float(np.infty)
        else:
            ni = self.n_visits[state_int]
            N  = sum([self.q_value[(state_int, a)]
                    for a in self.env.action_space])
            return self.value[state_int] +  2*math.sqrt(math.log(N)/ni)

    def init_stores(self):
        self.current_team = 0
        self.n_visits = {} # store number of visits
        self.value = {} # store the estimated value of a state
                        # all values are estimated from the point
                        # of view of team 1 (thus > 0 => team 1,
                        #                    and  < 0 => team 2)
        self.q_value = {}   # stores (state, action) pairs and their
                            # estimated values
        self.link = {}  # makes the link between hashed state and real state
        self.expanded = [] # list of expanded nodes
        self.children = {} # dict of state -> [(action, child_state)]
    
    




