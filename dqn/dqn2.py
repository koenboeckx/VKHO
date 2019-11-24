"""
https://keon.io/deep-q-learning/
"""
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import gym
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('dqn2')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

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

class GenericAgent:
    def __init__(self, env, **kwargs):
        self.obs_size  = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.gamma = kwargs.get('gamma', 0.95)
        hidden_size = kwargs.get('hidden_size', 24)
        
        self.epsilon = 1.0 # eps-greedy param for exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=2000)

        self.ex = kwargs.get('ex')
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        raise NotImplementedError

    def replay(self, batch_size):
        raise NotImplementedError

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class MLPAgent(GenericAgent):
    """Agent that implements learning with Multilayer Perceptron"""
    def __init__(self, env, **kwargs):
        super(MLPAgent, self).__init__(env, **kwargs)
        hidden_size = kwargs.get('hidden_size', 24)
        lr = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.model = MultiOutputRegressor(LinearRegression())

        # initialize random weights
        X = np.random.randn(self.batch_size, self.obs_size)
        y = np.random.randn(self.batch_size, self.n_actions)
        self.model.fit(X, y)

    def act(self, state):
        ""
        # probs = softmax(self.model.predict([state]))[0]
        # action = np.random.choice(range(self.n_actions), p=probs[0])
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            act_values = self.model.predict([state])
            action = np.argmax(act_values)
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            predicted = self.model.predict([state])[0][action]
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict([next_state]))
            self.model.fit()

        
class DQNAgent(GenericAgent):
    def __init__(self, env, **kwargs):
        super(DQNAgent, self).__init__(env, **kwargs)
        hidden_size = kwargs.get('hidden_size', 24)
        lr = kwargs.get('lr', 0.001)
        self.model = Net(self.obs_size, hidden_size, self.n_actions)
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=lr)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            act_values = self.model(torch.FloatTensor([state]))
            action = torch.argmax(act_values, dim=1)[0].item()
        return action 

    def replay(self, batch_size):
        self.optimizer.zero_grad()

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_v  = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        rewards_v = torch.FloatTensor(rewards)
        next_v    = torch.FloatTensor(next_states)
        done_mask = torch.ByteTensor(dones)

        q_vals_v = self.model(states_v)
        vals_pred = q_vals_v.gather(1, actions_v.unsqueeze(1)).squeeze(-1)
        next_q_vals = torch.max(self.model(next_v), dim=1)[0]
        vals_target = rewards_v + (1.-done_mask) * self.gamma * next_q_vals

        loss_v = F.mse_loss(vals_target, vals_pred)
        self.ex.log_scalar('loss', loss_v.item())
        loss_v.backward()
        self.optimizer.step()


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.ex.log_scalar('epsilon', self.epsilon)

@ex.config
def cfg():
    n_episodes = 5000
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    lr = 0.01
    hidden_size = 128
    batch_size = 256

@ex.automain
def run(n_episodes, batch_size, gamma, lr):
    env = gym.make('CartPole-v0')
    agent = MLPAgent(env, gamma=gamma, lr=lr,
                    batch_size=batch_size, ex=ex)

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