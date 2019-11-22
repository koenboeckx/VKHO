import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import gym

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('dqn2')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    n_episodes = 100
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    lr = 0.001
    hidden_size = 24
    batch_size = 32


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, **kwargs):
        obs_size  = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        hidden_size = kwargs.get('hidden_size', 24)
        self.gamma = kwargs.get('gamma', 0.95)
        lr = kwargs.get('lr', 0.001)
        self.model = Net(obs_size, hidden_size, self.n_actions)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        
        self.epsilon = 1.0 # eps-greedy param for exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=2000)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            act_values = self.model(torch.FloatTensor([state]))
            action = torch.argmax(act_values, dim=1)[0].item()
        return action 

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_v = torch.FloatTensor([state])
            next_state_v = torch.FloatTensor([next_state])
            target = torch.FloatTensor([reward])
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_v), dim=1)[0]
            target_f = self.model(state_v)

            self.optimizer.zero_grad()
            loss_v = F.mse_loss(target.squeeze(), target_f[0][action])
            loss_v.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


@ex.automain
def run(n_episodes, batch_size, gamma, lr):
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, gamma=gamma, lr=lr)

    for ep_idx in range(n_episodes):
        state = env.reset()
        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode {}/{}, score: {}'.format(ep_idx, n_episodes, time_t))
                ex.log_scalar('reward', time_t)
                break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)