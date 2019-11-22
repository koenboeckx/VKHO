import random
import copy
from collections import namedtuple

import gym
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('dqn')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

Experience = namedtuple('Experience', field_names = [
    'state', 'action', 'reward', 'next_state', 'done'
])

class DQNModel(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, n_hidden, n_actions, gamma, lr, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.model  = DQNModel(obs_size, n_hidden, n_actions).to(self.device)
        self.target = DQNModel(obs_size, n_hidden, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def get_action(self, state, epsilon=0.0):
        d = random.random()
        if d < epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            q_vals = self.model(torch.FloatTensor(state))
            action = torch.argmax(q_vals).item()
        return action

class ReplayBuffer:
    def __init__(self, maxlen=100):
        self.maxlen = maxlen
        self.content = []
    
    def __len__(self):
        return len(self.content)
    
    def insert(self, item):
        if len(self) == self.maxlen:
            self.content.pop(0)
        self.content.append(item)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.content), batch_size, replace=False)
        return [self.content[idx] for idx in indices]

def generate_episode(env, agent, eps=0.0):
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, epsilon=eps)
        next_state, reward, done, _  = env.step(action)
        episode.append(Experience(state, action, reward, next_state, done))
        state = next_state
    return episode

def generate_samples(env, agent, n_samples, eps=0.0):
    samples = []
    state = env.reset()
    for _ in range(n_samples):
        action = agent.get_action(state, epsilon=eps)
        next_state, reward, done, _  = env.step(action)
        samples.append(Experience(state, action, reward, next_state, done))
        state = next_state if not done else env.reset()
    return samples

def compute_loss(agent, batch):
    states, actions, rewards, next_states, dones = list(zip(*batch))
    states_v  = torch.FloatTensor(states).to(agent.device)
    actions_v = torch.LongTensor(actions).to(agent.device)
    rewards_v = torch.FloatTensor(rewards).to(agent.device)
    next_states_v = torch.FloatTensor(next_states).to(agent.device)
    done_mask = torch.ByteTensor(dones).to(agent.device)

    values_v  = agent.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values_v = torch.max(agent.target(next_states_v), dim=1)[0]
    next_state_values_v[done_mask] = 0.0
    target_values_v = rewards_v + agent.gamma * next_state_values_v
    
    loss_t = F.mse_loss(values_v, target_values_v)
    return loss_t


def train(env, agent, ex, n_steps=1000, buffer_size=32, batch_size=16, sync_rate=2000):
    buffer = ReplayBuffer(buffer_size)
    test_env = copy.deepcopy(env)
    state = env.reset()
    all_rewards = []
    for step_idx in range(n_steps):
        eps = max(0.05, 1. - step_idx/5000.)
        ex.log_scalar('epsilon', eps)
        action = agent.get_action(state, epsilon=eps)
        next_state, reward, done, _  = env.step(action)
        buffer.insert(Experience(state, action, reward, next_state, done))
        state = next_state if not done else env.reset()

        if len(buffer) < batch_size:
            continue

        batch = buffer.sample(batch_size)
        loss_t = compute_loss(agent, batch)
        ex.log_scalar('loss', loss_t.item())

        agent.optimizer.zero_grad()
        loss_t.backward()
        agent.optimizer.step()

        cur_reward = len(generate_episode(test_env, agent))
        ex.log_scalar('duration', cur_reward)

        all_rewards.append(cur_reward)
        if len(all_rewards) > 100:
            ex.log_scalar('duration100', sum(all_rewards[-100:])/100)

        if step_idx > 0 and step_idx % sync_rate == 0:
            agent.sync_models()

@ex.config
def cfg():
    rl_type = 'q_learning'
    gamma = 0.95
    lr = 0.005
    n_hidden = 128
    n_steps = 50000
    buffer_size = 32
    batch_size = 16
    sync_rate = 200

@ex.automain
def run(gamma, lr, n_hidden, n_steps, buffer_size, batch_size, sync_rate):
    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(obs_size, n_hidden, n_actions, gamma, lr, device)
    train(env, agent, ex, n_steps, buffer_size, batch_size, sync_rate)
    