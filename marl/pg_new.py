import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

from envs import Environment, all_actions
from collections import namedtuple
import random

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(self.conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x.float()).view(x.size()[0], -1)
        logits = self.policy(x)
        return logits

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
        input_shape = (1, params['board_size'], params['board_size'])
        self.model = A2CModel(input_shape, n_actions=8)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=params['learning_rate'])
    
    def get_action(self, state):
        with torch.no_grad():
            logits = self.model(preprocess([state]))
            action = Categorical(logits=logits).sample()
        return action.item()
    
    def discount_rewards(self, rewards):
        returns, R = [], 0.0
        own_rewards = [reward[self.idx] for reward in rewards]
        for reward in reversed(own_rewards):
            R = reward + params['gamma'] * R
            returns.insert(0, R)
        return returns

    def update(self, episode):
        self.optimizer.zero_grad()

        states, actions, rewards, _, dones = zip(*episode)
        logits_v = self.model(preprocess(states))
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.tensor(own_actions)
        
        returns_v = torch.tensor(self.discount_rewards(rewards))
        log_probs = F.log_softmax(logits_v, dim=1)
        log_prob_actions = log_probs[range(len(episode)), actions_v]
        loss = returns_v * log_prob_actions
        loss = -loss.mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()



params = {
    'n_episodes':       100,
    'board_size':       11,
    'gamma':            0.99,
    'learning_rate':    0.001,
}

def preprocess(states):
    """Process state to serve as input to convolutionel net."""
    bs = params['board_size']
    boards = np.zeros((len(states), 1, bs, bs))
    for idx, state in enumerate(states):
        board = np.array([int(b) for b in state.board])
        board = np.reshape(board, (1, bs, bs))
        boards[idx] = board
    return torch.tensor(boards)

def play_episode(env, agents):
    episode = []
    state = env.get_init_game_state()
    while True:
        actions = [agent.get_action(state) for agent in agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = env.terminal(next_state) != 0
        episode.append(Experience(state, actions, reward, next_state, done))
        if done:
            return episode
        state = next_state

def train(env, learners, opponents):
    loss = [None, ] * len(learners)
    agents = learners + opponents
    for epi_idx in range(params['n_episodes']):
        episode = play_episode(env, agents)
        for agent in learners:
            loss[agent.idx] = agent.update(episode)
        print(f"{epi_idx:5d}: len(episode) = {len(episode)}, loss0 = {loss[0]:8.5f}")
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learners  = [A2CAgent(idx, device) for idx in [0, 1]]
    opponents = [RandomTank(idx) for idx in [2, 3]]
    agents = learners + opponents

    env = Environment(agents, size=params['board_size'])
    train(env, learners, opponents)