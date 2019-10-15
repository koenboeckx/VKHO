from itertools import product
import random
import time
import numpy as np

from game.agents import Tank

DEBUG  = False
DEBUG2 = True


# TODO: fix this import error
#from envs import all_actions

#joint_actions =  dict(enumerate((product(all_actions.keys(),
#                                         all_actions.keys())))) 

joint_actions =  dict(enumerate(product(range(8),
                                         range(8))))

def state_to_int(state):
    """convert a state namedtuple to a unique integer.
    Added: include team that is to play from this state"""
    return hash(str(state))

class CommandedTank(Tank):
    """Tank that gets actions from a commander"""
    def __init__(self, idx):
        super(CommandedTank, self).__init__(idx)
    
    def get_action(self, obs):
        """Asks the commander to provide an action"""
        return self.commander.get_action(obs) # TODO: is this needed?

class Player:
    """Player that commands 2 tanks. Uses MCTS to search action space
    and provide best action for both tanks when requested."""
    def __init__(self, id, env):
        self.id = id # 0 or 1
        self.env = env
        self.visited  = {}
        self.n_visits = {}
        self.q_values = {}
        self.expanded = []
        self.action_space = joint_actions.keys()
        self.action_space_n = len(self.action_space)
    
    def is_leaf(self, state):
        return state_to_int(state) not in self.expanded
    
    def ucb(self, state):
        """Returns the list of UCB1 values for all actions player can
        take in 'state'."""
        state_int = state_to_int(state)
        ucb_vals = []

        N = sum(self.n_visits[state_int])
        for action in self.action_space:
            val = self.q_values[state_int][action]
            ni  = self.n_visits[state_int][action]
            if ni == 0: # action has never been performed in this state
                ucb_vals.append(float(np.infty))
            else:
                ucb = val + 2*np.sqrt(np.log(N)/ni)
                ucb_vals.append(ucb)
        return ucb_vals

    
    def pick_best_action(self, state):
        ucb_vals = self.ucb(state)
        _, best_action_int = max([(val, action) 
                                  for action, val in enumerate(ucb_vals)])
        return best_action_int
    
    def get_next(self, state, actions):
        if self.id == 0:
            actions = actions + (0, 0)
        else:
            actions = (0, 0) + actions
        next_state = self.env.sim_step(state, actions)
        return next_state


class MCTS:
    """Controls the MCTS search. Contains 2 players"""
    def __init__(self, player1, player2, **kwargs):
        self.players = (player1, player2)
        
        # for debugging
        player1.mcts = self
        player2.mcts = self

        self.max_search_time = kwargs.get('max_search_time', 2)

    def other(self, player):
        """Return the other player from self.players"""
        if player.id == 0:
            return self.players[1]
        else:
            return self.players[0]
    
    def get_action(self, player, state):
        state_int = state_to_int(state)
        if state_int not in player.n_visits:
            player.visited[state_int]  = False
            player.n_visits[state_int] = [0] * player.action_space_n
            player.q_values[state_int] = [0] * player.action_space_n
        start_time = time.time()
        while time.time() - start_time < self.max_search_time:
            self.one_iteration(player, state)
        
        return player.pick_best_action(state)
        
    def one_iteration(self, player, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""
        current_state = start_state
        current_player = player

        visited_nodes = [] # keep track of visited states and performed action in game tree

        while not current_player.is_leaf(current_state):
            if DEBUG:
                print('in tree traversal')
            best_action_idx = current_player.pick_best_action(current_state)
            best_action = joint_actions[best_action_idx]
            next_state = current_player.get_next(current_state, best_action)
            visited_nodes.append((current_state, best_action_idx))
            
            current_player = self.other(current_player)
            current_state = next_state
        
        #visited_nodes.append((current_state, None)) # add last state without action

        state_int = state_to_int(current_state)

        if not current_player.visited[state_int]: # first visit to this (already expanded) state
            current_player.visited[state_int] = True
            reward = self.rollout(current_state, current_player)
            # TODO: add backprop here?

        else: # node already visited => expand now
            if DEBUG2: print('expanding node {} for player {}'.format(state_to_int(current_state),
                                                                      current_player.id))
            for action_id in player.action_space:
                actions = joint_actions[action_id]
                child_state = current_player.get_next(current_state, actions)
                child_state_int = state_to_int(child_state)

                # add these nodes to visited nodes with initial values: ni=0, ti=0
                current_player.visited[state_int] = False # was state already visited?
                current_player.n_visits[state_int][action_id] = 0
                current_player.q_values[state_int][action_id] = 0 # TODO: does this work? -> YES
                
                # add child states to other player # TODO: is this the right way?
                other_player = self.other(current_player)
                other_player.visited[child_state_int] = False # was state already visited?
                other_player.n_visits[child_state_int] = [0] * other_player.action_space_n
                other_player.q_values[child_state_int] = [0] * other_player.action_space_n
            
            # add current node to list of expanded nodes
            current_player.expanded.append(state_int)

            # add last node to list of visited nodes
            #visited_nodes.append((current_state, action_id)) # ? do we want this?

            # make last expanded state the current state
            current_state = child_state
            current_player = self.other(current_player)
            
            # perform rollout from current state
            reward = self.rollout(current_state, current_player)
        
        self.backprop(visited_nodes, player, reward)
    
    def backprop(self, nodes, player, reward):
        if DEBUG:
            print('in backprop')
            print('nodes = ', nodes)

        for state, action in reversed(nodes):
            state_int = state_to_int(state)
            if DEBUG:
                print('Updating: ', state_int, action)
            player.n_visits[state_int][action] += 1
            player.q_values[state_int][action] += reward # TODO: need to store performed action

    
    def rollout(self, state, player):
        """Perform rollout (= random simulation), starting in 'state' until game is over."""
        if DEBUG: print('in rollout')
        while player.env.terminal_state(state) == 0: # no team has both players dead
            player = self.other(player)
            # generate random action
            action_idx = random.sample(joint_actions.keys(), 1)[0]
            action = joint_actions[action_idx]
            state = player.get_next(state, action)
            
            if DEBUG:
                #player.env.render(state.board)
                print('alive = ', state.alive)
                print('ammo  = ', state.ammo)
            
        if DEBUG:
            print('Game won by player {}'.format(player.id))
        #return player.env.terminal_state(state) # TODO: adapt env to be player agnostic
        return 1 if player.id == 0 else -1 # no need to analyze reward: if game ends when playermoves, he won


    
