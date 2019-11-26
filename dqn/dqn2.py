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

class GRUNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, n_layers=1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        
        # recurrent layer
        self.rnn = nn.GRU(input_size=obs_size,      # The number of expected features in the input x
                          hidden_size=hidden_size,  # The number of features in the hidden state h
                          num_layers=n_layers,      # Number of recurrent layers (default=1)
                          bias=True,                # Use bias weights
                          )
        # fully-connected layer
        self.fc = nn.Linear(self.hidden_dim, n_actions)

    def forward(self, x):
        batch_size = x.size(0)

        # initialize hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # passing in the input and hidden state into the model and obtain outputs
        out, hidden = self.rnn(x.unsqueeze(0), hidden)

        # reshape the outputs such that they can be fit into fully-connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        """This method generates the first hidden state of zeros which we'll use
        in the forward pass. We'll send the tensor holding the hidden state to
        the device we specified earlier as well."""
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class GenericAgent:
    def __init__(self, env, **kwargs):
        self.obs_size  = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.gamma = kwargs.get('gamma', 0.95)
        self.hidden_size = kwargs.get('hidden_size', 24)
        
        self.epsilon = kwargs.get('epsilon', 1.0) # eps-greedy param for exploration
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)

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
    """Agent that implements learning with Multilayer Perceptron.
    THIS DOESN'T WORK (yet)"""
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
        net = kwargs.get('net', Net)
        self.model = net(self.obs_size, hidden_size, self.n_actions)
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=lr)
        self.loss_min = np.infty # for debugging
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            self.model.eval()
            act_values = self.model(torch.FloatTensor([state]))
            action = torch.argmax(act_values, dim=1)[0].item()
        return action 

    def replay(self, batch_size):
        self.model.train()
        self.optimizer.zero_grad()

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_v  = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        rewards_v = torch.FloatTensor(rewards)
        next_v    = torch.FloatTensor(next_states)
        done_mask = torch.FloatTensor(dones)

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
        
        if loss_v.item() < self.loss_min:
            self.loss_min = loss_v.item()
        if loss_v.item() > 3*self.loss_min:
            print('loss too high')

        return loss_v.item()

@ex.config
def cfg():
    n_episodes = 1000
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999 # .995
    lr = 0.01
    hidden_size = 128
    batch_size = 16 # 512
    seed = 1234

"""
DEGUGGING Idea: 1. set fixed seed
                2. Execute run / store in MongoDB
                3. In Ombniboard: find index when loss.item() makes jumps
                4. Repeat run with same seed, set breakpoint at critical indexes
"""


@ex.automain
def run(n_episodes, batch_size, gamma, lr, epsilon_decay, seed):
    print('running...')
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, gamma=gamma, lr=lr,
                    batch_size=batch_size, ex=ex,
                    epsilon_decay=epsilon_decay,
                    net=Net)
    loss_min = np.infty

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
            loss = agent.replay(batch_size)
