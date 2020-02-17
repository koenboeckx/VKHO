# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import random, copy
import numpy as np
import gym
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from collections import namedtuple

# %%
from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('GYM_QL')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

# %%
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.content = []
    
    def __len__(self):
        return len(self.content)
    
    def insert(self, item):
        self.content.append(item)
        if len(self) > self.size:
            self.content.pop(0)
    
    def insert_list(self, items):
        for item in items:
            self.insert(item)
    
    def can_sample(self, N):
        return len(self) >= N
    
    def sample(self, N):
        assert self.can_sample(N)
        return random.sample(self.content, N)


# %%
Experience = namedtuple('Experience', field_names = [
    'state', 'action', 'reward', 'next_state', 'done'
])


# %%
def generate_episode(env, agent):
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        episode.append(Experience(state=state, action=action, reward=reward,
                                 next_state=next_state, done=done))
        state = next_state
    return episode


# %%
class Scheduler:
    def __init__(self, start, stop, decay=0.99):
        self.stop  = stop
        self.decay = decay
        self.value = start
    
    def __call__(self):
        self.value *= self.value * self.decay
        return max(self.value, self.stop)


# %%
class Agent(nn.Module):
    def __init__(self, env,  n_hidden, gamma, lr):
        super().__init__()
        self.env = env
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, env.action_space.n)
        )
        self.target = copy.deepcopy(self.net)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.epsilon_sched = Scheduler(start=1.0, stop=0.01, decay=0.99)
        self.gamma = gamma
    
    def sync(self):
        self.target.load_state_dict(self.net.state_dict())
        
    def forward(self, x):
        x = x.float()
        qvals = self.net(x)
        return qvals
    
    def act(self, state):
        epsilon = self.epsilon_sched()
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = self(torch.tensor([state]))[0]
                action = qvals.max(0)[1].item()
            return action
        
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_v = torch.tensor(states)
        current_q = self(states_v)[range(len(batch)), actions]
        rewards_v = torch.tensor(rewards)
        dones_v = torch.FloatTensor(dones)
        #set_trace()
        
        
        next_states_v = torch.tensor(next_states)
        next_q = self(next_states_v)
        next_qmax = next_q.max(1)[0]
        targets = rewards_v + self.gamma * (1.-dones_v) * next_qmax
        
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, targets.detach())
        loss.backward()
        self.optimizer.step()


# %%
N_STEPS = 5000

@ex.config
def cfg():
    n_hidden     = 24
    lr           = 0.001
    buffer_size  = 512
    batch_size   = 16
    gamma        = 0.99
    sync_rate    = 100


# %%
@ex.automain
def run(n_hidden, lr, buffer_size, batch_size, gamma, sync_rate):
    env    = gym.make('CartPole-v0')
    agent  = Agent(env, n_hidden=n_hidden, gamma=gamma, lr=lr)
    buffer = ReplayBuffer(size=buffer_size)

    for idx in range(N_STEPS):
        episode = generate_episode(env, agent)
        ex.log_scalar("epi_length", len(episode), step=idx)
        buffer.insert_list(episode)
        if not buffer.can_sample(batch_size):
            continue
        batch = buffer.sample(batch_size)
        agent.update(batch)
        if idx % sync_rate == 0:
            agent.sync()

