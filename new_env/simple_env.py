"""
Simple Environment: one-vs-one
"""

import random, copy
import numpy as np

all_actions = { 0: 'do_nothing',
                1: 'aim',  # prepare to fire on enemy
                2: 'fire',
                3: 'move_north',
                4: 'move_south',
                5: 'move_west',
                6: 'move_east'
}

class State:
    def __init__(self, agents, params):
        self.agents = agents
        self.position = {}
        self.alive = {}
        self.ammo  = {}
        self.aim   = {}
        taken = [] # avoids placing both players in same position
        for agent in self.agents:
            position = self.generate_position(params['board_size'])
            while position in taken:
                position = self.generate_position(params['board_size'])
            taken.append(position)
            self.position[agent] = position
            self.alive[agent] = True
            self.ammo[agent]  = params['init_ammo']  
            self.aim[agent]   = None
        # fixed initial positions
        #self.position[agents[0]] = (0, params['board_size']//2)
        #self.position[agents[1]] = (params['board_size']-1, params['board_size']//2)

    def __str__(self):
        s = f""
        for agent in self.agents:
            s += f"Agent {agent}: alive = {self.alive[agent]}, ammo = {self.ammo[agent]},"
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
    def __init__(self, state, agent):
        self.agent = agent
        self.own_position = state.position[agent]

        self.alive = state.alive[agent]
        self.ammo  = state.ammo[agent]
        self.aim   = state.aim[agent]
        for other in state.position:
            if other is not agent:
                self.other_position = state.position[other]
                self.other_alive    = state.alive[other]
    
    def __str__(self):
        s  = f"Observation for agent {self.agent}: "
        s += f"{'alive' if self.alive else 'dead'}, ammo = {self.ammo}, aim = {self.aim} @ postion {self.own_position};"
        s += f" opponent is {'alive' if self.other_alive else 'dead'} at position {self.other_position}"
        return s
    __repr__ = __str__

class Agent:
    def __init__(self, id, params):
        self.id = id
        self.team = 'blue' if id == 0 else 'red' # assign agents to team => adapt when more agents
        self.max_range = params['max_range']
    
    def __str__(self):
        return str(self.id)
    
    def set_env(self, env):
        self.env = env
    
    def act(self, obs):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in range(self.env.n_actions)
                        if action not in unavail_actions]
        return random.choice(avail_actions)
        #return 0

class SimpleEnvironment:
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
    
    def reset(self):
        self.state = State(self.agents, self.params)
        self.unavailable_actions = self.get_unavailable_actions()
        return self.state.copy()
    
    def get_state(self):
        return self.state
    
    def check_conditions(self, state, agent, action):
        assert agent in self.agents
        if state.alive[agent] is False: # no actions allowed for dead agent
                return False
        if   action == 0: # do_nothing
            return True # always allowed
        elif action == 1: # aim
            if state.aim[agent] is not None: # not allowed if already aiming
                return False
            else:
                return True
        elif action == 2: # fire
            if state.aim[agent] is None: # not allowed if not aiming
                return False
            elif state.ammo[agent] <= 0: # not allowed if no ammo remaining
                return False
            else:
                return True
        elif action == 3: # move_north
            if self.free(state.position[agent], 'north'):
                return True
            else:
                return False
        elif action == 4: # move_south
            if self.free(state.position[agent], 'south'):
                return True
            else:
                return False
        elif action == 5: # move_west
            if self.free(state.position[agent], 'west'):
                return True
            else:
                return False
        elif action == 6: # move_east
            if self.free(state.position[agent], 'east'):
                return True
            else:
                return False
        else:
            return False # action not in all_actions

    def get_new_position(self, position, direction):
        "returns new position if taken step in direction from position"
        if direction == 'north':
            new_position = position[0]-1, position[1]
        elif direction == 'south':
            new_position = position[0]+1, position[1]
        elif direction == 'east':
            new_position = position[0], position[1]+1
        elif direction == 'west':
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

    def other(self, agent):
        "return other agent"
        for other in self.agents:
            if other is not agent:
                return other
    
    def distance(self, agent1, agent2):
        x1, y1 = self.state.position[agent1]
        x2, y2 = self.state.position[agent2]
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def step(self, actions):
        "'actions' is dict of agent -> action pairs"
        assert len(actions) == len(self.agents)
        for agent, action in actions.items():
            if not self.check_conditions(self.state, agent, action):
                continue # if action not allowed, do nothing
            if action == 0: # do_nothing
                continue
            elif action == 1: # aim
                self.state.aim[agent] = self.other(agent)
            elif action == 2: # fire
                if self.distance(agent, self.other(agent)) < agent.max_range:
                    self.state.alive[self.other(agent)] = False
                self.state.aim[agent] = None    # lose aim after firing
                self.state.ammo[agent] -= 1     # decrease ammo after firing
            elif action in [3, 4, 5, 6]: # move north, south, west, east
                directions = {3: 'north', 4: 'south', 5: 'west', 6: 'east'}
                new_position = self.get_new_position(self.state.position[agent],
                                                     directions[action])
                self.state.position[agent] = new_position
        self.available_actions = self.get_unavailable_actions()

        rewards, done, info = self.get_rewards(self.state), self.terminal(self.state) is not False, {}
        return self.get_state(), rewards, done, info

    def terminal(self, state):
        "returns winning team ('red' or 'blue')"
        for agent in self.agents:
            if not state.alive[agent]:
                return 'blue' if agent.team == 'red' else 'red'
        if all([state.ammo[agent] == 0 for agent in self.agents]):
            return 'out-of-ammo'
        return False
    
    def get_rewards(self, state):
        terminal = self.terminal(state)
        if terminal == 'out-of-ammo': # because all out of ammo
            return {self.agents[0]: -1.,
                    self.agents[1]: -1.}
        if not terminal: # game not done => reward is penalty for making move
            return {self.agents[0]: -self.params['step_penalty'],
                    self.agents[1]: -self.params['step_penalty']}
        elif terminal == 'blue':
            return {self.agents[0]:  1.,
                    self.agents[1]: -1.}
        elif terminal == 'red':
            return {self.agents[0]: -1.,
                    self.agents[1]:  1.}
        else:
            raise ValueError(f'Unknown team {terminal}')

    def _get_available_actions(self, state): # no longer used
        available_actions = {}
        for agent in self.agents:
            available_actions[agent] = []
            for action in range(self.n_actions):
                if self.check_conditions(state, agent, action):
                    available_actions[agent].append(action)
        return available_actions
    
    def get_unavailable_actions(self):
        unavailable_actions = {}
        for agent in self.agents:
            unavailable_actions[agent] = []
            for action in range(self.n_actions):
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
        
#---------------------------------- test -------------------------------------
def test_step():
    agents = [Agent(0), Agent(1)]
    env = SimpleEnvironment(agents)
    state, done = env.reset(), False
    while not done:
        env.render()
        print(env.get_observation(agents[0]))
        actions = dict([(agent, agent.act(state)) for agent in env.agents])
        next_state, rewards, done, _ = env.step(actions)
        state = next_state
    env.render()
    print(env.get_observation(agents[0]))
    print(env.get_observation(agents[1]))

if __name__ == '__main__':
    test_step()
