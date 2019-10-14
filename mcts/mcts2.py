from itertools import product
import random

DEBUG = True


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

class Player:
    """Player that commands 2 tanks. Uses MCTS to search action space
    and provide best action for both tanks when requested."""
    def __init__(self, id, env, agent1, agent2):
        self.id = id # 0 or 1
        self.env = env
        self.agents = (agent1, agent2)
        self.n_visits = {}
        self.q_values = {}
        self.expanded = []
        self.action_space = joint_actions.keys()
    
    def is_leaf(self, state):
        return state_to_int(state) not in self.expanded
    
    def pick_best_action(self, state):
        vals = self.q_values[state_to_int(state)]
        _, best_action_int = max([(val, action) 
                                  for action, val in enumerate(vals)])
        return joint_actions[best_action_int]
    
    def get_next(self, state, actions):
        if self.id == 0:
            actions = actions + (0, 0)
        else:
            actions = (0, 0) + actions
        next_state = self.env.sim_step(state, self.id, actions)
        return next_state


class MCTS:
    """Controls the MCTS search. Contains 2 players"""
    def __init__(self, player1, player2, **kwargs):
        self.players = (player1, player2)
        self.max_search_time = kwargs.get('max_search_time', 10)

    def other(self, player):
        """Return the other player from self.players"""
        if player.id == 0:
            return self.players[1]
        else:
            return self.players[2]
    
    def one_iteration(self, player, start_state):
        """Runs one iteration of MCTS for 'player', starting
        in state 'start_state'"""
        current_state = start_state
        current_player = player

        visited_nodes = [] # keep track of visited states and performed action in game tree

        while not current_player.is_leaf(current_state):
            best_action = current_player.pick_best_action(current_state)
            next_state = current_player.get_next(current_state, best_action)
            visited_nodes.append((current_state, best_action))
            
            current_player = self.other(current_player)
            current_state = next_state

        state_int = state_to_int(current_state)

        if current_player.n_visits[state_int] == 0: # first visit to this state
            reward = self.rollout(current_state, current_player)
            # TODO: add backprop here?
        else: # node already visited => expand now
            for actions in player.action_space:
                child_state = current_player.get_next(current_state, actions)
                child_state_int = state_to_int(child_state)

                # add these nodes with initial values: ni=0, ti=0
                current_player.n_visits[child_state_int] = 0
                current_player.q_values[state_int][actions] = 0 # TODO: does this work?
            
            # add current node to list of expanded nodes
            current_player.expanded.append(state_int)
            # make last expanded state the current state
            current_state = child_state
            current_player = self.other(current_player)
            visited_nodes.append(current_state)

            # perform rollout from current state
            reward = self.rollout(current_state, current_player)
        
        self.backprop(visited_nodes, player, reward)
    
    def backprop(self, nodes, player, reward):
        for state, action in reversed(nodes):
            state_int = state_int(state)
            player.n_visits[state_int] += 1
            player.q_values[state_int][action] += reward # TODO: need to store performed action
    
    def rollout(self, state, player):
        """Perform rollout (= random simulation), starting in 'state' until game is over."""
        while player.env.terminal_state(state) == 0: # no team has both players dead
            player = self.other(player)
            # generate random action
            action = random.sample(joint_actions.keys(), 1)[0] 
            state = player.env.sim_step(state, action)
            if DEBUG:
                player.env.render(state.board)
                print('alive = ', state.alive)
                print('ammo  = ', state.ammo)
        return player.env.terminal_state(state) # TODO: adapt env to be player agnostic


    
