"""
[13Feb20] More complex Environment: two-vs-two
[25Mar20] Add visibility information to state
"""

import random, copy, pickle
import numpy as np
from collections import namedtuple

import torch

Action = namedtuple('Action', field_names = ['id', 'name', 'target'])

def generate_terrain(size):
    """returns list with position of obstacles.
    An obstacle blocks both passage and visibility"""
    terrain = []
    terrain.append((size//2-1, size//2-1))
    terrain.append((size//2, size//2-1))
    terrain.append((size//2-1, size//2))
    terrain.append((size//2, size//2))
    return terrain


class State:
    def __init__(self, agents, args):
        self.agents = agents
        self.position = {}
        self.alive = {}
        self.ammo  = {}
        self.aim   = {}
        taken = [] # avoids placing players in same position
        for agent in self.agents:
            position = self.generate_position(args.board_size)
            while position in taken:
                position = self.generate_position(args.board_size)
            taken.append(position)
            self.position[agent] = position

        if args.fixed_init_position: # set fixed initial position (only use for 1v1 and 2v2)
            if args.n_friends == 1:
                self.position[agents[0]] = (args.board_size//2, 0)
                self.position[agents[1]] = (args.board_size//2, args.board_size-1)
            elif args.n_friends == 2:
                self.position[agents[0]] = (args.board_size//2-1, 0)
                self.position[agents[1]] = (args.board_size//2+1, 0)
                self.position[agents[2]] = (args.board_size//2-1, args.board_size-1)
                self.position[agents[3]] = (args.board_size//2+1, args.board_size-1)

        for agent in self.agents:
            self.alive[agent] = True
            self.ammo[agent]  = args.init_ammo
            self.aim[agent]   = None
        
        self.terrain = generate_terrain(args.board_size)
        self.visible = dict()

    def __str__(self):
        s = f""
        for agent in self.agents:
            s += f"Agent {agent} (team = {agent.team}): alive = {self.alive[agent]}, ammo = {self.ammo[agent]},"
            s += f" aim = {self.aim[agent]}"
            s += f" Position: {self.position[agent]}\n"
        return s
    __repr__ = __str__

    def generate_position(self, board_size):
        position = (random.randint(0, board_size-1),
                    random.randint(0, board_size-1))
        return position

class Observation: 
    """An Observation has 3 components:
        (1) own data: positions, alive, ammo, aim
        (2) friends:  relative position of friendly (living) forces in self.friends
        (3) enemies:  relative position of enemy (living) forces in self.enemies
    """
    # TODO: add visibility indicator for enemies
    def __init__(self, state, agent):
        self.agent = agent
        self.own_position = state.position[agent]

        self.alive = state.alive[agent]
        self.ammo  = state.ammo[agent]
        self.aim   = state.aim[agent]

        self.friends = []
        self.enemies = []
        self.enemy_visibility = []
        own_pos = self.own_position
        for other in state.position:
            if other is not agent:
                other_pos = state.position[other]
                rel_pos = other_pos[0]-own_pos[0], other_pos[1]-own_pos[1]
                if other.team == agent.team:    # same team -> friends
                    if not state.alive[other]:
                        self.friends.append(False)
                    else:
                        self.friends.append(rel_pos)
                else:                           # other team -> enemies
                    if not state.alive[other]:
                        self.enemies.append(False)
                    else:
                        self.enemies.append(rel_pos)    
                    self.enemy_visibility.append(state.visible[(agent, other)])  
        
    def __str__(self):
        s  = f"Observation for agent {self.agent}: "
        s += f"{'alive' if self.alive else 'dead'}, ammo = {self.ammo}, aim = {self.aim} @ postion {self.own_position}\n"
        s += f"     friends at {[pos for pos in self.friends]}\n"
        s += f"     enemies at {[pos for pos in self.enemies]}"
        return s
    __repr__ = __str__

class Agent:
    def __init__(self, id, team):
        self.id = id
        self.team = team # "blue" or "red"
        
    
    def __str__(self):
        return str(self.id)
    #__repr__ = __str__
    
    def set_env(self, env):
        self.env = env
        self.actions = self.generate_actions()
        self.n_actions = len(self.actions)
        self.max_range = env.args.max_range # defined here to use args from env
 
    def generate_actions(self):
        actions = [
            Action(0, 'do_nothing', target=None),
            Action(1, 'fire', target=None),
            Action(2, 'move_north', target=None),
            Action(3, 'move_south', target=None),
            Action(4, 'move_west', target=None),
            Action(5, 'move_east', target=None),
        ]
        enemies = [agent for agent in self.env.agents if agent.team != self.team]
        for id, enemy in enumerate(enemies):
            actions.append(Action(id + 6, 'aim', enemy))
        return actions
  
    def act(self, obs, **kwargs):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions if action not in unavail_actions]
        return random.choice(avail_actions)
        #return 0
    
    def set_hidden_state(self):
        pass

    def get_hidden_state(self):
        if hasattr(self, 'hidden_state'):
            return self.hidden_state
        else:
            return torch.zeros((1, self.env.args.n_hidden)) # changed from None because of issue with torch.stack() in qmix2.py
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

class Environment:
    def __init__(self, agents, args):
        self.args = args
        self.register_agents(agents)
        self.state = State(self.agents, args)
        self.board_size = args.board_size
        self.terrain = generate_terrain(self.board_size)
    
    def register_agents(self, agents):
        self.agents = agents
        for agent in self.agents:
            agent.set_env(self)
        self.teams = {
            'blue':  [agent for agent in agents if agent.team == 'blue'],
            'red':   [agent for agent in agents if agent.team == 'red'],
        }

    def reset(self):
        self.state = State(self.agents, self.args)
        self.unavailable_actions = self.get_unavailable_actions()
        return self.state
    
    def get_state(self):
        return self.state
    
    def check_conditions(self, state, agent, action):
        assert agent in self.agents
        if action.name == 'do_nothing':
            return True # always allowed
        elif state.alive[agent] is False: # no actions (except 'do_nothing') allowed for dead agent
            return False
        elif action.name == 'fire':
            if state.aim[agent] is None: # not allowed if not aiming
                return False
            elif state.ammo[agent] <= 0: # not allowed if no ammo remaining
                return False
            else:
                return True
        elif action.name in ['move_north', 'move_south', 'move_west', 'move_east']:
            if self.free(state.position[agent], action.name):
                return True
            else:
                return False
        elif action.name == 'aim':
            if state.aim[agent] is action.target: # not allowed if already aiming
                return False
            else:
                return True
        else:
            return False # action not in all_actions

    def get_new_position(self, position, direction):
        "returns new position if taken step in direction from position"
        if direction == 'move_north':
            new_position = position[0]-1, position[1]
        elif direction == 'move_south':
            new_position = position[0]+1, position[1]
        elif direction == 'move_east':
            new_position = position[0], position[1]+1
        elif direction == 'move_west':
            new_position = position[0], position[1]-1 
        return new_position
    
    def free(self, position, direction):
        "check if position in 'direction' from 'position' is free"
        new_position = self.get_new_position(position, direction)
        # 1. check if new position is still on the board
        if new_position[0] < 0 or new_position[0] >= self.board_size:
            return False
        if new_position[1] < 0 or new_position[1] >= self.board_size:
            return False
        # 2. check  if nobody else occupies the new position
        for agent in self.agents:
            if self.state.position[agent] == new_position:
                return False
        return True

    def distance(self, agent1, agent2):
        x1, y1 = self.state.position[agent1]
        x2, y2 = self.state.position[agent2]
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def act(self, observations):
        actions = {}
        for agent in self.agents:
            actions[agent] = agent.act(observations[agent])
        return actions

    def step(self, actions):
        "'actions' is dict of agent -> action pairs"
        assert len(actions) == len(self.agents)
        for agent in self.agents:
            action = actions[agent]
            if not self.check_conditions(self.state, agent, action):
                continue # if action not allowed, do nothing
            if action.name == 'do_nothing':
                continue
            elif action.name == 'aim':
                self.state.aim[agent] = action.target
            elif action.name == 'fire':
                opponent = self.state.aim[agent]
                if self.distance(agent, opponent) < agent.max_range:
                    self.state.alive[opponent] = False
                self.state.aim[agent] = None    # lose aim after firing
                self.state.ammo[agent] -= 1     # decrease ammo after firing
            elif action.name in ['move_north', 'move_south', 'move_west', 'move_east']:
                new_position = self.get_new_position(self.state.position[agent], action.name)
                self.state.position[agent] = new_position
        self.available_actions = self.get_unavailable_actions()

        rewards, done, info = self.get_rewards(self.state), self.terminal(self.state) is not False, {}
        return self.get_state(), rewards, done, info

    def terminal(self, state):
        "returns winning team ('red' or 'blue')"
        if all([state.alive[agent] == False for agent in self.teams["red"]]):      # all "red"s are dead
            return "blue"
        elif all([state.alive[agent] == False for agent in self.teams["blue"]]):   # all "blue"s are dead
            return "red"
        if all([state.ammo[agent] == 0 for agent in self.agents if state.alive[agent]]):
            return 'out-of-ammo'
        return False
    
    def get_rewards_(self, state): # TODO: return reward per team, not per agent
        terminal = self.terminal(state)
        if terminal == 'out-of-ammo': # because all out of ammo
            rewards = {}
            for agent in self.agents:
                rewards[agent] = -1.
        elif not terminal: # game not done => reward is penalty for making move
            rewards = {}
            for agent in self.agents:
                rewards[agent] = -args.step_penalty
        elif terminal == 'blue':
            rewards = {}
            for agent in self.teams["blue"]:
                rewards[agent] =  1.0
            for agent in self.teams["red"]:
                rewards[agent] = -1.0
        elif terminal == 'red':
            rewards = {}
            for agent in self.teams["blue"]:
                rewards[agent] = -1.0
            for agent in self.teams["red"]:
                rewards[agent] =  1.0
        else:
            raise ValueError(f'Unknown team {terminal}')
        return rewards
    
    def get_rewards(self, state):
        terminal = self.terminal(state)
        if terminal == 'out-of-ammo': # because all out of ammo
            rewards = {'blue': -1, 'red': -1}
        elif not terminal: # game not done => reward is penalty for making move
            rewards = {'blue': -self.args.step_penalty,
                       'red':  -self.args.step_penalty}
        elif terminal == 'blue':
            rewards = {'blue': 1, 'red': -1}
        elif terminal == 'red':
            rewards = {'blue': -1, 'red': 1}
        else:
            raise ValueError(f'Unknown team {terminal}')
        return rewards

    def get_unavailable_actions(self):
        unavailable_actions = {}
        for agent in self.agents:
            unavailable_actions[agent] = []
            for action in agent.actions:
                if not self.check_conditions(self.state, agent, action):
                    unavailable_actions[agent].append(action)
        return unavailable_actions

    def render(self, state=None):
        if state is None:
            state = self.state
        arr = np.zeros((self.board_size, self.board_size)) - 1
        for agent in self.agents:
            x, y = state.position[agent]
            arr[x, y] = agent.id
        s = '_' * (self.board_size + 2) + '\n'
        for obstacle in self.state.terrain:
            x, y = obstacle
            arr[x, y] = -2
        for x in range(self.board_size):
            s += '|'
            for y in range(self.board_size):
                if arr[x, y] == -1:
                    s += '.'
                elif arr[x, y] == -2:
                    s += 'x'
                else:
                    s += str(int(arr[x, y]))
            s += '|\n'
        s += '-' * (self.board_size + 2)
        print(s)
        print(state)
    
    def get_observation(self, agent):
        return Observation(self.state, agent)
    
    def get_all_observations(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_observation(agent)
        return observations

class RestrictedEnvironment(Environment):
    """Additional restriction: opponent must be visible when FIRING
    Only impacts method `check_conditions"""
    def __init__(self, agents, args):
        super().__init__(agents, args)
        self.visibility_dict = self._generate_vis_dict()
        self.set_state_visibility()
    
    def _generate_vis_dict(self):
        """
        Pre-compute the straight lines between all two points.
        :return: a dict with keys = ((xi, yi), (xj, yj)) pairs
                and values = [..., (xk, yk), ...] element along
                line between (xi, yi) and (xj, yj).
        """
        size = self.board_size
        N = 20 # number of points considered, 100 = randomly chosen value
        los = {}
        all_squares = [(x, y) for x in range(size) for y in range(size)]
        for (xi, yi) in all_squares:
            for (xj, yj) in all_squares:
                los[((xi, yi), (xj, yj))] = []
                dx = (xj - xi)/N
                dy = (yj - yi)/N
                xs = [xi+k*dx for k in range(N)]
                ys = []
                for k, x in zip(range(N), xs): # create list of consecutive, non-integer x values
                    if x == xi: # move in horizontal line
                        ys.append(yi + k*dy)
                    else:
                        ys.append(yi + (yj - yi) * (x - xi) /(xj - xi))
                for x, y in zip(xs, ys):
                    los[((xi, yi), (xj, yj))].append((int(round(x)), int(round(y))))

                los[((xi, yi), (xj, yj))] = list(set(los[((xi, yi), (xj, yj))])) # remove doubles
                # remove endpoints:
                if (xi, yi) in los[((xi, yi), (xj, yj))]:
                    los[((xi, yi), (xj, yj))].remove((xi, yi))
                if (xj, yj) in los[((xi, yi), (xj, yj))]:
                    los[((xi, yi), (xj, yj))].remove((xj, yj))

        return los

    def set_state_visibility(self):
        # add visibility
        for agent in self.agents:
            for other in self.agents:
                if agent.team == other.team: # agens belong to same team
                    continue
                else:
                    self.state.visible[(agent, other)] = self.visible(agent, other)

    def step(self, actions):
        "extend .step method of parent class"
        self.state, rewards, done, info = super().step(actions)
        self.set_state_visibility()
        return self.get_state(), rewards, done, info
    
    def reset(self):
        "extend .reset method of parent class"
        self.state = super().reset()
        self.set_state_visibility()
        return self.state

    def visible(self, agent, opponent):
        x0, y0 = self.state.position[agent]
        x1, y1 = self.state.position[opponent]
        los = self.visibility_dict[(x0, y0), (x1, y1)] # Line-Of-Sight dictionary between two points
        # check agents
        for other_agent in self.agents:
            other_pos = self.state.position[other_agent]
            if other_pos in los:
                return False
        # check terrain obstacles
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x,y) in self.state.terrain: # means: obstacle blocks visibility in terrain
                    if (x, y) in los:
                        return False
        return True

    def free(self, position, direction):
        """check if position in 'direction' from 'position' is free
        subclassing to take terrain into account"""
        new_position = self.get_new_position(position, direction)
        # 1. check if new position is still on the board
        if new_position[0] < 0 or new_position[0] >= self.board_size:
            return False
        if new_position[1] < 0 or new_position[1] >= self.board_size:
            return False
        # 2. check  if nobody else occupies the new position
        for agent in self.agents:
            if self.state.position[agent] == new_position:
                return False
        # 3. check if nothing blocks new position in terrain
        if new_position in self.state.terrain:
            return False
        return True

    def check_conditions(self, state, agent, action):
        assert agent in self.agents
        if action.name == 'do_nothing':
            return True # always allowed
        elif state.alive[agent] is False: # no actions (except 'do_nothing') allowed for dead agent
            return False
        elif action.name == 'fire':
            if state.aim[agent] is None: # not allowed if not aiming
                return False
            elif state.ammo[agent] <= 0: # not allowed if no ammo remaining
                return False
            elif not self.visible(agent, state.aim[agent]):
                return False
            else:
                return True
        elif action.name in ['move_north', 'move_south', 'move_west', 'move_east']:
            if self.free(state.position[agent], action.name):
                return True
            else:
                return False
        elif action.name == 'aim':
            if state.aim[agent] is action.target: # not allowed if already aiming
                return False
            else:
                return True
        else:
            return False # action not in all_actions

#---------------------------------- test -------------------------------------
def test_step():
    from settings import args
    team_blue = [Agent(0, "blue"), Agent(1, "blue")]
    team_red  = [Agent(2, "red"),  Agent(3, "red")]
    agents = team_blue + team_red
    env = Environment(agents, args)
    state, done = env.reset(), False
    i = 0
    while not done:
        print(f"step {i}")
        env.render()
        print(env.get_observation(agents[0]))
        print('Unavaible for agent 0:\n', env.get_unavailable_actions()[agents[0]])

        actions = dict([(agent, agent.act(state)) for agent in env.agents])
        print(actions)
        next_state, rewards, done, _ = env.step(actions)
        state = next_state
        i += 1
        if i > 100:
            print('stop')
    env.render()
    print(env.get_observation(agents[0]))
    print(env.get_observation(agents[2]))
    print(rewards.values())
    print(f'Terminated in {i} steps')

def test_visibility():
    from settings import args
    team_blue = [Agent(0, "blue"), Agent(1, "blue")]
    team_red  = [Agent(2, "red"),  Agent(3, "red")]
    agents = team_blue + team_red
    env = RestrictedEnvironment(agents, args)
        
    agent0 = env.agents[0]
    agent1 = env.agents[1]
    agent3 = env.agents[3]
    env.render()
    print(env.visible(agent1, agent3)) # -> True

    observations = env.get_all_observations()

    env.state.position[agent0] = (4,4)
    env.render()
    env.visible(agent1, agent3)
    print(env.visible(agent1, agent3)) # -> False

def test_terrain():
    print(generate_terrain(10))

    

if __name__ == '__main__':
    test_visibility()