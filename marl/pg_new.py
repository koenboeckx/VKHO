import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

from envs import Environment, all_actions
from collections import namedtuple
import random

import torch
from torch import nn
from torch.nn import functional as F

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        pass

    def forward(self, x):
        pass

class Tank:
    def __init__(self, idx):
        super(Tank, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = 500
        self.max_range = 5
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action 
    
    def __repr__(self):
        return self.type + str(self.idx)

class RandomTank(Tank):
    def __init__(self, idx):
        super(RandomTank, self).__init__(idx)
    
    def get_action(self, obs):
        return random.randint(0, 7)

class A2CAgent(Tank):
    def __init__(self, idx, device):
        super().__init__(idx)
        self.model = A2CModel(params['board_size'], n_actions=8)
    
    def get_action(self, obs):
        return random.randint(0, 7)
    
    def update(self, episode):
        pass

params = {
    'board_size':       11,
}

def play_episode(env, agents):
    episode = []
    state = env.get_init_game_state()
    while True:
        actions = [agent.get_action(state) for agent in agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = env.terminal(next_state) != 0
        if done:
            print('done')
        episode.append(Experience(state, actions, reward, next_state, done))
        if done:
            return episode
        state = next_state

def train(env, learners, opponents):
    agents = learners + opponents
    episode = play_episode(env, agents)
    for exp in episode:
        print([all_actions[a] for a in exp.actions])
        print(exp.next_state)
        print('______________')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learners  = [A2CAgent(idx, device) for idx in [0, 1]]
    opponents = [RandomTank(idx) for idx in [2, 3]]
    agents = learners + opponents

    env = Environment(agents, size=params['board_size'])
    train(env, learners, opponents)