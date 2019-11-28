"""Test of IQL with gym_env"""

import random
from collections import deque

import gym
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_iql_gym')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, x):
        return self.fc(x)

class GymAgent:
    def __init__(self, buffer_size, gamma):
        self.idx = 0

        self.buffer = deque(maxlen=buffer_size)
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
    
    def __repr__(self):
        return 'gym0'

    def set_model(self, input_shape, n_actions, n_hidden, lr):
        self.model = Net(input_shape, n_hidden, n_actions)
        self.target = Net(input_shape, n_hidden, n_actions)
        self.sync_models()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
   
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            values = self.model(torch.FloatTensor([state]))
            return torch.argmax(values).item()
    
    def remember(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batchsize):
        batch = random.sample(self.buffer, batchsize)
        return batch

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = list(zip(*batch))

        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        rewards_v = torch.FloatTensor(rewards)
        next_states_v = torch.FloatTensor(next_states)
        done_mask = torch.ByteTensor(dones)

        predictions_v = self.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_vals_v = torch.max(self.target(next_states_v), dim=1)[0]
        next_vals_v = next_vals_v.detach() # change compared to iql.py
        next_vals_v[done_mask] = 0.0
        targets_v = rewards_v + self.gamma * next_vals_v

        loss = F.mse_loss(targets_v, predictions_v)
        return loss


@ex.config
def cfg():
    n_episodes = 1000
    sync_rate  = 250
    n_hidden = 64
    lr = 0.05
    buffer_size = 500
    batch_size = 8
    gamma = 0.99

@ex.automain
def train(n_episodes, n_hidden, lr, buffer_size, batch_size, gamma, sync_rate, maxsteps=500):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = maxsteps-1

    n_state   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = GymAgent(buffer_size, gamma)
    agent.set_model(n_state, n_actions, n_hidden, lr)

    for episode_idx in range(n_episodes):
        state = env.reset()
        for time_t in range(maxsteps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember((state, action, reward, next_state, done))
            if done:
                break
        ex.log_scalar('reward', time_t)
    
        if len(agent.buffer) > batch_size:
            batch = agent.sample(batch_size)
            loss = agent.compute_loss(batch)
            ex.log_scalar('loss', loss.item())

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            ex.log_scalar('epsilon', agent.epsilon)
        
        if (episode_idx + 1) % sync_rate == 0:
            agent.sync_models()



