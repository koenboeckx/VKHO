from itertools import product
import random
import time
import pickle
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

def max_random(val_actions):
    """Returns maximum value. If multiple: select randomly."""
    items = [val for val, action in val_actions]
    max_val = max(items)
    idxs = [idx for idx, val in enumerate(items) if val == max_val]
    rand_idx = random.sample(idxs, 1)[0]
    return val_actions[rand_idx]


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
        #return True # TODO: remove to activate function
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

        N = self.n_visits[state]
        if N == 0:
            print_state(state)
            raise ValueError('Division by zero')
        for child in self.children[state]:
            val = self.v_values[child]
            ni  = self.n_visits[child]
            if ni == 0: # this action has never been performed in this state
                uct = float(np.infty)
            else:
                uct = val/ni + 2*np.sqrt(np.log(N)/ni) # TODO: how to handle this in minimax?
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
            best_action_idx = self.pick_best_action(state, visited_nodes) # force next state to be a new state
            best_action = joint_actions[best_action_idx]
            next_state = self.get_next(state, best_action)
            visited_nodes.append(state)
                       
            state = next_state
        
        visited_nodes.append(state) # add last state without action
        return state, visited_nodes

    def one_iteration(self, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""

        # find first leaf node while descending the game tree
        current_state, visited_nodes = self.find_leaf(start_state)

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
        for state in reversed(nodes):
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
            
        return -1 if state.player == 0 else 1 # if game ends when player moved, he won
                                              # BUT: state.player == player to move! thus other player won
                                              # TODO: check if this reasoning is correct
    
    def save(self, filename):
        with open(filename, "wb" ) as file:
            pickle.dump([self.n_visits, self.v_values, self.children], file)
        
    def load(self, filename):
        with open(filename, "rb" ) as file:
            stores = pickle.load(file)
            self.n_visits, self.v_values, self.children = stores
            

# TODO: adapt MCTS so that selection is done by other player
# mcts1 = MCTS(player=1)
# mcts2 = MCTS(player=2)
# mcts1.set_opponent(mcts2)
# mcts2.set_opponent(mcts1)            
def  play_game(env, filename=None):
    """Play a single game"""
    mcts_stores = (MCTS(env), MCTS(env))

    state = env.get_init_game_state()
    result = env.terminal(state)
    while result == 0: # nobody has won
        current_player = state.player
        action_idx = mcts_stores[current_player].get_action(state)
        action = joint_actions[action_idx]
        print('Player {} plays ({}, {}) - # visited nodes = {}'.format(
            current_player, all_actions[action[0]],
            all_actions[action[1]], len(mcts_stores[current_player].n_visits)))
    print('UCT for state = ', sorted(mcts_stores[current_player].uct(state),
         reverse=True))

    state = mcts_stores[current_player].get_next(state, action)
    env.render(state)
    print_state(state)

    result = env.terminal(state)