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
        self.max_search_time = 10 # max search time in seconds => proxy for search depth

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
        counter = 0 # for debugging; TODO: remove later
        start_time = time.time()
        while time.time() - start_time < self.max_search_time:
        #for i in range(2): # remove this line later
            print('Iteration = {}'.format(counter))
            self.one_iteration(team, state)
            counter += 1
        
        # now select the best action (based on UCB value TODO: check alternatives)
        best_action, _ = self.pick_best_action(state, team)
        
        return best_action

    def pick_best_action(self, current_state, team):
        # construct list of successor states
        child_nodes = [self.env.sim_step(current_state, team, actions)
                          for actions in self.action_space]
        
        # compute UCB and select best action
        best = max if team == 0 else min
        _, best_action, next_state = best(list(zip(self.ucb(child_nodes), self.action_space, child_nodes)))
        return best_action, next_state

    def one_iteration(self, team, start_node):
        current_state = start_node
        visited_nodes = [current_state] # keep track of nodes visited along the way to perform backprop
        while not self.is_leaf(current_state):
            best_action, current_state = self.pick_best_action(current_state, team)
            # current_state = self.env.sim_step(current_state, best_action) # TODO: normally not needed
            visited_nodes.append(current_state)
            # switch teams after each round
            team = 0 if team == 1 else 1 
        
        state_int = state_to_int(current_state)

        if self.n_visits[state_int] == 0: # first visit to this node => perform rollout and backprop
            reward = self.rollout(current_state, team)
            self.backprop(visited_nodes, team, reward)
            
        else:
            # expand node -> no longer a leaf node
            for action in self.action_space:
                child_state = self.env.sim_step(current_state, team, action)
                child_state_int = state_to_int(child_state)
                # add this child state to visited nodes with ni = 0 and value = 0
                self.n_visits[child_state_int] = 0
                self.value[child_state_int] = 0
            # add current node to list of expanded nodes
            self.expanded.append(state_int)
            # make last expanded state the current state (why last? -> convenience)
            current_state = child_state
            visited_nodes.append(current_state)

            # perform rollout and backprop on last created node = current state
            reward = self.rollout(current_state, team)
            self.backprop(visited_nodes, team, reward)
            

    def backprop(self, visited_nodes, team, reward):
        """Propagate the reward signal back trough the list of visited nodes"""
        # TODO: verify if team really has no importance here
        for node in reversed(visited_nodes):
            node_int = state_to_int(node)
            self.n_visits[node_int] += 1
            self.value[node_int] += reward

    def rollout(self, state, team):
        """Perform rollout (= random simulation), starting in 'state' until game is over."""
        while self.env.terminal_state(state) == 0: # no team has both players dead
            team = 0 if team == 1 else 1 # switch teams after each round
            # generate random action
            action = (random.randint(0, self.env.n_actions),
                      random.randint(0, self.env.n_actions))
            state = self.env.sim_step(state, team, action)
            if DEBUG:
                self.env.render(state.board)
                print('alive = ', state.alive)
                print('ammo  = ', state.ammo)
        return self.env.terminal_state(state)
    
    def is_leaf(self, state):
        return state_to_int(state) not in self.expanded 

    def ucb(self, child_nodes):
        """Returns a list of UCB1 values for all children in child_nodes."""
        child_nodes_int = [state_to_int(child) for child in child_nodes]
        N = sum([self.n_visits[child] for child in child_nodes_int])
        ucb_vals = []
        for child_int in child_nodes_int:
            if self.n_visits[child_int] == 0:
                 ucb_vals.append(float(np.infty))
            else:
                ni = self.n_visits[child_int]
                ucb_vals.append(self.value[child_int] +  2*math.sqrt(math.log(N)/ni))
        return ucb_vals

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
    
    




