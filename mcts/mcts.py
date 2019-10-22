"""
Goal of this branch: rewrite mcts such that one MCTS
instance represents one player.
"""

from itertools import product
import random
import time
import pickle
import numpy as np

#from game.envs import print_state, all_actions

# TODO: fix this import error
#from envs import all_actions

#joint_actions =  dict(enumerate((product(all_actions.keys(),
#                                         all_actions.keys())))) 

joint_actions =  dict(enumerate(product(range(8),
                                         range(8))))

def max_random(val_actions):
    """Returns maximum value. If multiple: select randomly."""
    items = [val for val, action in val_actions]
    max_val = max(items)
    idxs = [idx for idx, val in enumerate(items) if val == max_val]
    rand_idx = random.sample(idxs, 1)[0]
    return val_actions[rand_idx]

class MCTSStore:
    """Container object for the different stores of an MCTS player"""
    def __init__(self):
        self.n_visits = {} # store visits to each child node (thus for each action)
        self.v_values = {} # store value for each child node (thus for each action)

class MCTS:
    """Controls the MCTS search. One instance per player"""
    def __init__(self, stores, env, **kwargs): # TODO: adapt this class
        """
        Main class to manage MCTS process.
        :param: stores: tuples of MCTSStore instance
                env:    Environment instance
        """
        self.stores = stores
        self.env = env
        self.args = kwargs
        self.max_search_time = kwargs.get('max_search_time', 2)

        self.children = {} # maintain list of parent state -> [child states]

        self.action_space = joint_actions.keys()
        self.action_space_n = len(self.action_space)
    
    def other(self, player):
        return 1 if player == 0 else 0
   
    def is_leaf(self, state):
        return state not in self.children # state hasn't been expanded
    
    def get_action(self, state):
        p_idx = state.player 
        if state not in self.stores[p_idx].n_visits:
            self.stores[p_idx].n_visits[state] = [0] * self.action_space_n
            self.stores[p_idx].v_values[state] = [0.0] * self.action_space_n
        
        start_time = time.time()

        while time.time() - start_time < self.max_search_time:
        #for i in range(20):
            self.one_iteration(state)
        
        return self.pick_best_action(state)
    
    def check_actions(self, state, action):
        """Check if joint action (0..63) is allowed in state.
        Uses env.check_condtion(.). Goal: remove actions that
        produce no 'new' children."""
        #return True # remove to activate function
        actions = joint_actions[action]
        if state.player == 0:
            for agent, action in zip([0, 1], actions):
                if not self.env.check_conditions(state, agent, action):
                    return False
        else: # player 1
            for agent, action in zip([2, 3], actions):
                if not self.env.check_conditions(state, agent, action):
                    return False
        return True
    
    def pick_best_action(self, state, visited_nodes=[]):
        uct_vals = self.uct(state)

        vals_actions = [(val, action) for action, val in enumerate(uct_vals)]
        # filter away all actions that are not allowed:
        vals_actions = list(filter(lambda va: self.check_actions(state, va[1]), vals_actions))
        # filter away all actions that lead to previously visited states (avoids infinite loops):
        vals_actions = list(filter(lambda va: self.children[state][va[1]] not in visited_nodes, vals_actions)) 
        # pick now the best value-action pair (and discard value)
        #vals_actions = [(val, action) for action, val in enumerate(uct_vals)
        #                if self.check_actions(state, action)                    # filter away all actions that are not allowed
        #                and self.children[state][action] not in visited_nodes   # filter away all actions that lead to previously visited states (avoids infinite loops)
        #                ]

        # pick now the best value-action pair (and discard value)
        try:
            _, best_action_idx = max(vals_actions)
        except ValueError:
            print('ValueError: max() arg is an empty sequence')
            pass
        return best_action_idx
    
    def uct(self, state):
        """Returns the list of UCT (UCB applied to Trees) values for all actions player can
        take in 'state'.
        :params: state: state for which all UCT values are computed (one per child (action))
        :return: list of uct values (one per potential action)
        """
        uct_vals = []

        N = sum(self.stores[state.player].n_visits[state])
        if N == 0:
            raise ValueError('Division by zero')
        for ni, val in zip(self.stores[state.player].n_visits[state],
                           self.stores[state.player].v_values[state]):
            if ni == 0: # this action has never been performed in this state
                uct = float(np.infty)
            else:
                uct = val/ni + 2*np.sqrt(np.log(N)/ni)
            uct_vals.append(uct)
        return uct_vals
    
    def get_next(self, state, actions):
        """
        Get next state from environment, by performing action in state.
        Action is performed for state.player = current player.
        :params: state:   current state, instance of State
                 actions: tuple (action_1, action_2) with action_i in [0..7]
        :return: next state
        """
        if state.player == 0:
            actions = actions + (0, 0)
        else:
            actions = (0, 0) + actions
        next_state = self.env.step(state, actions)
        return next_state

    def find_leaf(self, state):
        """
        Traverse existing game tree, each time picking node with highest UCT for allowed action,
        until a non-expanded node (= leaf node) is found.
        :params: state : State tuple from wich the search starts
        :return: (state, visited_nodes): leaf state, nodes encoutered during search, for backprop.
        """
        visited_nodes = [] # keep track of visited states and performed action in game tree
        # TODO: take into account terminal states
        while not self.is_leaf(state): # walk through existing game tree until leaf
            best_action_idx = self.pick_best_action(state, [v for v, _ in visited_nodes]) # force next state to be a new state
            best_action = joint_actions[best_action_idx]
            next_state = self.get_next(state, best_action)
            visited_nodes.append((state, best_action_idx))
                       
            state = next_state
        
        #visited_nodes.append((state, best_action_idx)) # add last state without action
        return state, visited_nodes

    def one_iteration(self, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""

        # find first leaf node while descending the game tree
        current_state, visited_nodes = self.find_leaf(start_state)
        current_player = current_state.player

        if sum(self.stores[current_player].n_visits[current_state]) == 0: # first visit to this (already expanded) state
            reward = self.rollout(current_player, current_state)
            visited_nodes.append((current_state, 0)) # choose arbitrarily first action to update
            self.backprop(current_player, visited_nodes, reward)

        else: # node already visited => expand now
            self.children[current_state] = []
            for action_idx in self.action_space:
                actions = joint_actions[action_idx]
                child_state = self.get_next(current_state, actions)

                # add these nodes to visited nodes with initial values: ni=0, ti=0 if not already present
                # remark: this is now for the other player
                self.children[current_state].append(child_state)
                other_player = 1 - current_player
                if child_state not in self.stores[other_player].n_visits:
                    self.stores[other_player].n_visits[child_state] = [0] * self.action_space_n
                    self.stores[other_player].v_values[child_state] = [0.0] * self.action_space_n  

            # make last expanded state the current state
            current_state = child_state
            
            # perform rollout from current state
            reward = self.rollout(current_player, current_state)
            self.backprop(current_player, visited_nodes, reward)
    
    def backprop(self, player, nodes, reward):
        """Perform backup of reward over nodes in 'nodes' list.
        :param: player: player that initiated the rollout
                nodes: list of nodes visited during selection phase
                reward: numerical reward obtained during rollout from viewpoint of player
        """
        for state, action in reversed(nodes):
            self.stores[state.player].n_visits[state][action] += 1 # TODO: state.player instead of player
            self.stores[state.player].v_values[state][action] += reward
            
            reward = -reward
            player = 1 if player == 0 else 0
    
    def rollout(self, current_player, state):
        """Perform rollout (= random simulation), starting in 'state' until game is over.
        :param: current_player: player that initiated the rollout
                state: state form wich rollout starts
        :return: numerical reward signal
        """
        while self.env.terminal(state) == 0: # no team has both players dead
            # generate random action
            action_idx = random.sample(joint_actions.keys(), 1)[0]
            action = joint_actions[action_idx]
            state = self.get_next(state, action)
            
        return 1 if state.player == current_player else -1
    
    def save(self, filename):
        with open(filename, "wb" ) as file:
            pickle.dump([self.n_visits, self.v_values, self.children], file)
        
    def load(self, filename):
        with open(filename, "rb" ) as file:
            stores = pickle.load(file)
            self.n_visits, self.v_values, self.children = stores
            
         
