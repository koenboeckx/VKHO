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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, x):
        return self.fc(x)

class GymAgent:
    def __init__(self, buffer_size, gamma, epsilon_decay):
        self.idx = 0

        self.buffer = deque(maxlen=buffer_size)
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
    
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
            with torch.no_grad():
                values = self.model(torch.FloatTensor([state]))
                return torch.argmax(values).item()
    
    def remember(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batchsize):
        batch = random.sample(self.buffer, batchsize)
        return batch

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        rewards_v = torch.FloatTensor(rewards)
        next_states_v = torch.FloatTensor(next_states)
        done_mask = torch.ByteTensor(dones)

        predictions_v = self.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_vals_v = self.target(next_states_v).detach().max(1)[0]
        #next_vals_v = self.model(next_states_v).detach().max(1)[0]
        next_vals_v[done_mask] = 0.0
        targets_v = rewards_v + self.gamma * next_vals_v

        loss = F.mse_loss(targets_v, predictions_v)
        #loss = F.smooth_l1_loss(targets_v, predictions_v)
        return loss


@ex.config
def cfg():
    n_episodes = 10000
    sync_rate  = 1000
    n_hidden = 256
    lr = 0.001
    buffer_size = 1000000
    batch_size = 20
    gamma = 0.
    epsilon_decay = 0.995

@ex.automain
def train(n_episodes, n_hidden, lr, buffer_size, batch_size, gamma, sync_rate, epsilon_decay, maxsteps=500):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = maxsteps-1

    n_state   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = GymAgent(buffer_size, gamma, epsilon_decay)
    agent.set_model(n_state, n_actions, n_hidden, lr)
    print(agent.model)

    for episode_idx in range(n_episodes):
        state = env.reset()
        for time_t in range(maxsteps):
            if (episode_idx + 1) % 50 == 0:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            #if done:
            #    reward = -1
            agent.remember((state, action, reward, next_state, done))
            if done:
                break
    
            if len(agent.buffer) < batch_size: # don't train when not enough samples for batch
                continue
            
            # learning after each step in the episode
            batch = agent.sample(batch_size)
            loss = agent.compute_loss(batch)
            ex.log_scalar('loss', loss.item())

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        ex.log_scalar('reward', time_t)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        ex.log_scalar('epsilon', agent.epsilon)
        
        if (episode_idx + 1) % sync_rate == 0:
            agent.sync_models()




