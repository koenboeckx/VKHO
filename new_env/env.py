"""
[13Feb20] More complex Environment: two-vs-two
"""

import random, copy, pickle
import numpy as np
from collections import namedtuple

Action = namedtuple('Action', field_names = ['id', 'name', 'target'])

all_actions = { 0: 'do_nothing',
                1: 'aim0', # aim at agent 0 of enemy
                2: 'aim1', # aim at agent 1 of enemy
                3: 'fire',
                4: 'move_north',
                5: 'move_south',
                6: 'move_west',
                7: 'move_east'
}

class State:
    def __init__(self, agents, params):
        self.agents = agents
        self.position = {}
        self.alive = {}
        self.ammo  = {}
        self.aim   = {}
        taken = [] # avoids placing players in same position
        for agent in self.agents:
            position = self.generate_position(params['board_size'])
            while position in taken:
                position = self.generate_position(params['board_size'])
            taken.append(position)

            self.position[agent] = position
            self.alive[agent] = True
            self.ammo[agent]  = params['init_ammo']  
            self.aim[agent]   = None

    def __str__(self):
        s = f""
        for agent in self.agents:
            s += f"Agent {agent} (team = {agent.team}): alive = {self.alive[agent]}, ammo = {self.ammo[agent]},"
            s += f" aim = {self.aim[agent]}"
            s += f" Position: {self.position[agent]}\n"
        return s
    __repr__ = __str__

    def copy(self):
        return copy.deepcopy(self)

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
    def __init__(self, state, agent):
        self.agent = agent
        self.own_position = state.position[agent]

        self.alive = state.alive[agent]
        self.ammo  = state.ammo[agent]
        self.aim   = state.aim[agent]

        self.friends = []
        self.enemies = []
        own_pos = self.own_position
        for other in state.position:
            if other is not agent:
                other_pos = state.position[other]
                rel_pos = other_pos[0]-own_pos[0], other_pos[1]-own_pos[1]
                if other.team == agent.team: # same team -> friends
                    if not state.alive[other]:
                        self.friends.append(False)
                    else:
                        self.friends.append(rel_pos)
                else:
                    if not state.alive[other]:
                        self.enemies.append(False)
                    else:
                        self.enemies.append(rel_pos)

    def __str__(self):
        s  = f"Observation for agent {self.agent}: "
        s += f"{'alive' if self.alive else 'dead'}, ammo = {self.ammo}, aim = {self.aim} @ postion {self.own_position}\n"
        s += f"     friends at {[pos for pos in self.friends]}\n"
        s += f"     enemies at {[pos for pos in self.enemies]}"
        return s
    __repr__ = __str__

class Agent:
    def __init__(self, id, team, params):
        self.id = id
        self.team = team # "blue" or "red"
        self.max_range = params['max_range']
    
    def __str__(self):
        return str(self.id)
    
    def set_env(self, env):
        self.env = env
        self.actions = self.generate_actions()
        self.n_actions = len(self.actions)
 
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
  
    def act(self, obs):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions if action not in unavail_actions]
        return random.choice(avail_actions)
        #return 0
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

class Environment:
    def __init__(self, agents, params):
        self.register_agents(agents)
        self.actions = all_actions.copy()
        self.n_actions = len(self.actions)
        self.state = State(self.agents, params)
        self.board_size = params["board_size"]
        self.params = params
    
    def register_agents(self, agents):
        self.agents = agents
        for agent in self.agents:
            agent.set_env(self)
        self.teams = {
            'blue':  [agent for agent in agents if agent.team == 'blue'],
            'red':   [agent for agent in agents if agent.team == 'red'],
        }

    def reset(self):
        self.state = State(self.agents, self.params)
        self.unavailable_actions = self.get_unavailable_actions()
        return self.state.copy()
    
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
        elif action.name in ['move north', 'move_south', 'move_west', 'move_east']:
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
            elif action.name in ['move north', 'move_south', 'move_west', 'move_east']:
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
        if all([state.ammo[agent] == 0 for agent in self.agents]):
            return 'out-of-ammo'
        return False
    
    def get_rewards(self, state):
        terminal = self.terminal(state)
        if terminal == 'out-of-ammo': # because all out of ammo
            rewards = {}
            for agent in self.agents:
                rewards[agent] = -1.
        elif not terminal: # game not done => reward is penalty for making move
            rewards = {}
            for agent in self.agents:
                rewards[agent] = -self.params['step_penalty']
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
        arr = np.zeros((self.board_size, self.board_size))
        for agent in self.agents:
            pos = state.position[agent]
            arr[pos[0], pos[1]] = agent.id + 1
        print(arr)
        print(state)
    
    def get_observation(self, agent):
        return Observation(self.state, agent)
    
    def get_all_observations(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_observation(agent)
        return observations

#---------------------------------- test -------------------------------------
def test_step():
    params = {
        "max_range":    100,
        "board_size":   9,
        "init_ammo":    5,
        "step_penalty": 0.01,
    }
    team_blue = [Agent(0, "blue", params), Agent(1, "blue", params)]
    team_red  = [Agent(2, "red", params),  Agent(3, "red", params), Agent(4, "red", params)]
    agents = team_blue + team_red
    env = Environment(agents, params)
    state, done = env.reset(), False
    while not done:
        env.render()
        print(env.get_observation(agents[0]))
        actions = dict([(agent, agent.act(state)) for agent in env.agents])
        next_state, rewards, done, _ = env.step(actions)
        state = next_state
    env.render()
    print(env.get_observation(agents[0]))
    print(env.get_observation(agents[2]))
    print(rewards.values())

if __name__ == '__main__':
    test_step()