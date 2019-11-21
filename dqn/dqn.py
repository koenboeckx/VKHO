import random
import copy
from collections import namedtuple

import gym

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
    def __init__(self, input_shape, n_hidden, n_actions):
        super(DQNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.tensor(x).float()
        else:
            x = x.float()
        return self.fc(x)

class DQNAgent:
    def __init__(self, n_actions, gamma, lr, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.model  = DQNModel(4, 32, n_actions).to(self.device)
        self.target = DQNModel(4, 32, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def get_action(self, state, epsilon=0.0):
        d = random.random()
        if d < epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            q_vals = self.model([state])
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
    
    def sample(self, n):
        return random.sample(self.content, n)

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
    states_v  = torch.tensor(states).to(agent.device)
    actions_v = torch.LongTensor(actions).to(agent.device)
    rewards_v = torch.tensor(rewards).to(agent.device)
    next_states_v = torch.tensor(next_states).to(agent.device)
    dones_v = torch.ByteTensor(dones).float().to(agent.device)

    values_v  = agent.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    targets_v = rewards_v
    targets_v += (1. - dones_v) * agent.gamma * torch.max(agent.target(next_states_v), dim=1)[0]
    #targets_v += (1. - dones_v) * agent.gamma * torch.max(agent.model(next_states_v), dim=1)[0] # !! TODO: wrong 

    loss_t = F.mse_loss(values_v, targets_v)
    return loss_t


def train(env, agent, ex, n_steps=1000, buffer_size=32, batch_size=16, sync_rate=2000):
    buffer = ReplayBuffer(buffer_size)
    test_env = copy.deepcopy(env)
    state = env.reset()
    for step_idx in range(n_steps):
        action = agent.get_action(state, epsilon=0.1) # TODO: change fxed epsilon
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

        ex.log_scalar('duration', len(generate_episode(test_env, agent)))

        if step_idx > 0 and step_idx % sync_rate == 0:
            agent.sync_models()

@ex.config
def cfg():
    rl_type = 'q_learning'
    gamma = 0.99
    lr = 0.001
    n_steps = 10000
    buffer_size = 32
    batch_size = 16
    sync_rate = 2000

@ex.automain
def run(gamma, lr, n_steps, buffer_size, batch_size, sync_rate):
    env = gym.make('CartPole-v0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.action_space.n, gamma, lr, device)
    
    train(env, agent, ex, n_steps, buffer_size, batch_size, sync_rate)
    