from itertools import product
import random
import time
import numpy as np

from game.agents import Tank

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
    
    def get_action(self, player, state):
        if state not in self.n_visits:
            self.n_visits[state] = 0
            self.v_values[state] = 0
        
        start_time = time.time()

        while time.time() - start_time < self.max_search_time:
            self.one_iteration(player, state)
        
        return self.pick_best_action(player, state)
    
    def pick_best_action(self, player, state):
        ucb_vals = self.ucb(state)
        if player == 0: # pick highest UCB
            _, best_action_idx = max([(val, action) 
                                    for action, val in enumerate(ucb_vals)])
        elif player == 1: # pick lowest UCB; TODO: quid variance?
            _, best_action_idx = min([(val, action) 
                                    for action, val in enumerate(ucb_vals)])
        return best_action_idx
    
    def ucb(self, state):
        """Returns the list of UCB1 values for all actions player can
        take in 'state'."""
        ucb_vals = []

        N = self.n_visits[state]
        for child in self.children[state]:
            val = self.v_values[child]
            ni  = self.n_visits[child]
            if ni == 0: # action has never been performed in this state
                ucb_vals.append(float(np.infty))
            else:
                ucb = val + 2*np.sqrt(np.log(N)/ni)
                ucb_vals.append(ucb)
        return ucb_vals
    
    def get_next(self, player, state, actions):
        if player == 0:
            actions = actions + (0, 0)
        else:
            actions = (0, 0) + actions
        next_state = self.env.step(state, actions)
        return next_state
        
    def one_iteration(self, player, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""
        current_state = start_state
        current_player = player

        visited_nodes = [] # keep track of visited states and performed action in game tree

        while not self.is_leaf(current_state):
            best_action_idx = self.pick_best_action(current_player, current_state)
            best_action = joint_actions[best_action_idx]
            next_state = self.get_next(player, current_state, best_action)
            visited_nodes.append((current_state, player, best_action_idx))
            print('len(visited_nodes) = ', len(visited_nodes))
            
            current_player = self.other(current_player)
            current_state = next_state
        
        visited_nodes.append((current_state, player, None)) # add last state without action

        if self.n_visits[current_state] == 0: # first visit to this (already expanded) state
            reward = self.rollout(current_state, current_player)
            # TODO: add backprop here?

        else: # node already visited => expand now
            self.children[current_state] = []
            for action_id in player.action_space:
                actions = joint_actions[action_id]
                child_state = self.get_next(current_state, current_player, actions)

                # add these nodes to visited nodes with initial values: ni=0, ti=0
                self.children[current_state].append(child_state)
                self.n_visits[child_state] = 0
                self.v_values[child_state] = 0      

            # make last expanded state the current state
            current_state = child_state
            current_player = self.other(current_player)
            
            # perform rollout from current state
            reward = self.rollout(current_state, current_player)
        
        self.backprop(visited_nodes, player, reward)
    
    def backprop(self, nodes, player, reward):
        """Perform backup of reward over nodes in 'nodes' list.""" 
        for state, player, action in reversed(nodes):
            self.n_visits[state] += 1
            self.v_values[state] += reward
    
    def rollout(self, state, player):
        """Perform rollout (= random simulation), starting in 'state' until game is over."""
        while self.env.terminal(state) == 0: # no team has both players dead
            # generate random action
            action_idx = random.sample(joint_actions.keys(), 1)[0]
            action = joint_actions[action_idx]
            state = self.get_next(player, state, action)
            player = self.other(player)
            
        if DEBUG:
            print('Game won by player {}'.format(player.id))
        return 1 if player == 0 else -1 # if game ends when player moved, he won


    
