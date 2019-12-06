"""
Implement Counterfactual Multi-Agent Policy Gradients
https://arxiv.org/abs/1705.08926
"""

import collections
import random

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 

class Actor(nn.Module):
    def __init__(self, input_shape, gru_input_size, hidden_size, n_actions):
        super(Actor, self).__init__()

        self.input_fc = nn.Sequential( # takes as input: (observation of agent i,
            nn.Linear(input_shape[0], gru_input_size), #  advantage a,  
            nn.ReLU()                                  #  previous actions u_t-1)
        )
        self.rnn = nn.GRU(gru_input_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, n_actions)
    
    def _init_hidden(self):
        pass

    def foward(self, x):
        x = self.input_fc(x)
        pass

class Agent:
    def __init__(self):
        pass
    
    def sample_action(self, logits):
        probs = np.exp(logits)
        probs = probs / sum(probs)
        action_space = range(len(logits))
        return np.random.choice(action_space, p=probs)

class Buffer:
    def __init__(self, maxlen=100):
        self.capacity = maxlen
        self.contents = []
    
    def __len__(self):
        return len(self.contents)
    
    def insert(self, item):
        if len(self) > self.capacity:
            self.contents.pop(0)
        self.contents.append(item)
    
    def empty(self):
        self.contents = []
    
    def sample(self, N):
        if N > len(self):
            raise ValueError('Buffer is too small (len(buffer = {}), N = {})'.format(len(self), N))
        return random.sample(self.contents, N)


def train(n_training_episodes, buffersize):                         # TODO: for now, observation = full state
    buffer = Buffer(maxlen=buffersize)
    for train_episode in range(n_training_episodes):
        buffer.empty()
        # 1. populate buffer with experiences through interaction with environment
        for ec in range(batch_size):
            state = env.reset()
            t, done = 0, False
            for agent in agents:
                agent.init_hidden()
            while not done and t < TMAX:
                t = t + 1
                actions = [None,]*len(agents)
                for idx, agent in enumerate(agents):
                    h = agent.actor(state, agent.h_old, agent.u_old)
                    u = agent.sample(h)
                    actions[idx] = u
                    agent.h_old, agent.u_old = h, u
                next_state, reward, done, _ = env.step(actions)
                buffer.insert((state, actions, reward, next_state, done))

        # 2. compute TD targets y
