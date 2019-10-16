from itertools import product
import random
import time
import numpy as np

from game.agents import Tank
from game.envs import print_state

DEBUG  = False
DEBUG2 = True
DEBUG3 = True


# TODO: fix this import error
#from envs import all_actions

#joint_actions =  dict(enumerate((product(all_actions.keys(),
#                                         all_actions.keys())))) 

joint_actions =  dict(enumerate(product(range(8),
                                         range(8))))

class MCTS:
    """Controls the MCTS search. Contains 2 players"""
    def __init__(self, env, **kwargs):
        
        self.env = env
        self.args = kwargs
        self.max_search_time = kwargs.get('max_search_time', 2)

        self.n_visits = {}
        self.v_values = {}
        self.children = {}

        self.action_space = joint_actions.keys()
        self.action_space_n = len(self.action_space)

    def other(self, player):
        """Return the other player from self.players"""
        return 1 if player == 0 else 0
    
    def is_leaf(self, state):
        return state not in self.children # state hasn't been expanded
    
    def get_action(self, state):
        if state not in self.n_visits:
            self.n_visits[state] = 0
            self.v_values[state] = 0
        
        start_time = time.time()

        while time.time() - start_time < self.max_search_time:
            self.one_iteration(state)
        
        return self.pick_best_action(state)
    
    def check_actions(self, state, action):
        """Check if joint action (0..63) is allowed in state.
        Uses env.check_condtion(.). Goal: remove actions that
        produce no 'new' children."""
        actions = joint_actions[action]
        if state.player == 1:
            for agent, action in zip([0, 1], actions):
                if not self.env.check_conditions(state, agent, action):
                    return False
        else: # player 1
            for agent, action in zip([2, 3], actions):
                if not self.env.check_conditions(state, agent, action):
                    return False
        return True
    
    def pick_best_action(self, state):
        ucb_vals = self.ucb(state)
        # TODO: remove actions that are not allowed 
        _, best_action_idx = max([(val, action) 
                                    for action, val in enumerate(ucb_vals)
                                    if self.check_actions(state, action)])                                   
        return best_action_idx
    
    def ucb(self, state):
        """Returns the list of UCB1 values for all actions player can
        take in 'state'."""
        ucb_vals = []

        N = self.n_visits[state]
        if N == 0:
            print_state(state)
            raise ValueError('Division by zero')
        for child in self.children[state]:
            val = self.v_values[child]
            ni  = self.n_visits[child]
            if ni == 0: # this action has never been performed in this state
                ucb = float(np.infty)
            else:
                ucb = val + 2*np.sqrt(np.log(N)/ni)
            ucb_vals.append(ucb)
        return ucb_vals
    
    def get_next(self, state, actions):
        if state.player == 0:
            actions = actions + (0, 0)
        else:
            actions = (0, 0) + actions
        next_state = self.env.step(state, actions)
        return next_state
        
    def one_iteration(self, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""
        current_state = start_state

        visited_nodes = [] # keep track of visited states and performed action in game tree

        # TODO: take into account terminal states
        while not self.is_leaf(current_state): # walk through existing game tree until leaf
            best_action_idx = self.pick_best_action(current_state)
            best_action = joint_actions[best_action_idx]
            next_state = self.get_next(current_state, best_action)
            visited_nodes.append((current_state, best_action_idx))
            if len(visited_nodes) > 100:
                print('...loop...')
            
            current_state = next_state
        
        visited_nodes.append((current_state, None)) # add last state without action
        #print('len(visited_nodes) = ', len(visited_nodes))

        if self.n_visits[current_state] == 0: # first visit to this (already expanded) state
            reward = self.rollout(current_state)
            self.backprop(visited_nodes, reward)

        else: # node already visited => expand now
            self.children[current_state] = []
            for action_idx in self.action_space:
                actions = joint_actions[action_idx]
                child_state = self.get_next(current_state, actions)

                # add these nodes to visited nodes with initial values: ni=0, ti=0 if not already present
                self.children[current_state].append(child_state)
                if child_state not in self.n_visits:
                    self.n_visits[child_state] = 0
                    self.v_values[child_state] = 0      

            # make last expanded state the current state
            current_state = child_state
            
            # perform rollout from current state
            reward = self.rollout(current_state)
            self.backprop(visited_nodes, reward)
    
    def backprop(self, nodes, reward):
        """Perform backup of reward over nodes in 'nodes' list.""" 
        for state, action in reversed(nodes):
            self.n_visits[state] += 1
            if state.player == 0:
                self.v_values[state] += reward
            if state.player == 1:
                self.v_values[state] -= reward
    
    def rollout(self, state):
        """Perform rollout (= random simulation), starting in 'state' until game is over."""
        while self.env.terminal(state) == 0: # no team has both players dead
            # generate random action
            action_idx = random.sample(joint_actions.keys(), 1)[0]
            action = joint_actions[action_idx]
            state = self.get_next(state, action)
            
        return 1 if state.player == 0 else -1 # if game ends when player moved, he won


    
