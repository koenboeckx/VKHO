import collections, random
from itertools import count

import numpy as np
import gym
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

ALPHA = 0.7

params = {
    'env_name':             'CartPole-v0',
    'learning_rate':        0.001,
    'n_steps':              10000,
    'buffer_size':          500,
    'batch_size':           10,
    'gamma':                0.99,
    'stop_length':          198,
    'episodes_to_train':    100,
    'eval_rate':            100,
    'entropy_coeff':        0.01,
}

class Actor(nn.Module):
    def __init__(self, obs_shape, n_actions, n_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        logits = self.net(x.float())
        return logits

class Critic(nn.Module):
    def __init__(self, obs_shape, n_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        values = self.net(x.float())
        return values

class Agent:
    def __init__(self, obs_shape, n_actions):
        self.n_actions = n_actions
        self.actor = Actor(obs_shape, n_actions)
        self.critic = Critic(obs_shape)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), params['learning_rate'])
        self.critic_optimizer = optim.Adam(self.actor.parameters(), params['learning_rate'])
    
    def choose_action(self, observation):
        with torch.no_grad():
            logits = self.actor(torch.tensor(observation))
            action = Categorical(logits=logits).sample().item()
        return action
    
    def compute_returns(self, next_value, episode):
        returns = []
        R = next_value
        for exp in reversed(episode):
            mask = 0.0 if exp.done else 1.0
            R = params['gamma'] * R * mask + exp.reward
            returns.insert(0, R)
        return returns
    
    def update(self, episode):
        states, actions, rewards, next_states, dones = zip(*episode)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        next_state = torch.tensor(next_states[-1])
        next_value = self.critic(next_state).detach().item()
        returns = self.compute_returns(next_value, episode)

        states_v = torch.tensor(states)
        actions_v = torch.tensor(actions)
        returns_v = torch.tensor(returns).detach()

        logits_v = self.actor(states_v)
        values_v = self.critic(states_v)

        advantages = returns_v - values_v.squeeze()

        # policy loss
        log_probs = F.log_softmax(logits_v, dim=1)
        log_prob_actions = log_probs[range(len(episode)), actions_v]
        loss_pol = log_prob_actions * advantages.detach()
        loss_pol = -loss_pol.mean() # used to be .sum()

        loss_val = advantages.pow(2).mean()
        loss_val.backward()
        self.critic_optimizer.step()
        
        loss_pol.backward()
        self.actor_optimizer.step()

        ## compute statistics
        grad_max, grad_means, grad_counts = 0.0, 0.0, 0
        for p in self.actor.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means  += (p.grad ** 2).mean().sqrt().item()
            grad_counts += 1

        statistics = {
            'loss_pol':     loss_pol.item(),
            'loss_val':     loss_val.item(),
            'grad_max':     grad_max,
            'grad_l2':      grad_means/grad_counts,
        }

        return statistics
    
    def play_episode(self, env, show=False):
        state, done = env.reset(), False
        while not done:
            if show:
                env.render()
            action = self.choose_action(state)
            state, _, done, _ = env.step(action)
    

Experience = collections.namedtuple('Experience', 
                field_names = ['state', 'action', 'reward', 'next_state', 'done'])

class Buffer:
    def __init__(self, size):
        self.size = size
        self.content = []
    
    def __len__(self):
        return len(self.content)

    def insert(self, item):
        if isinstance(item, list):
            self.content.extend(item)
        else:
            self.content.append(item)
        while len(self) > self.size:
            self.content.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.content, batch_size)

class Source:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.state = env.reset()
        self.done  = False
    
    def get_steps(self, n):
        experiences = []
        for _ in range(n):
            action = self.agent.choose_action(self.state)
            next_state, reward, self.done, _ = self.env.step(action)
            experiences.append(Experience(self.state, action, reward, next_state, self.done))
            self.state = self.env.reset() if self.done else next_state
        return experiences

def eval(agent, test_env, n=20):
    total_length = 0
    for _ in range(n):
        state, done = test_env.reset(), False
        while not done:
            action = agent.choose_action(state)
            state, _, done, _ = test_env.step(action)
            total_length += 1
    return total_length/n

def train(env, test_env):
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.n)
    run_length = 10
    for idx in range(params['n_steps']):
        batch = []
        state = env.reset()
        for i in count():
            if idx % 100 == 0: env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            batch.append(Experience(state, action, reward, next_state, done))
            if done:
                run_length = ALPHA * run_length + (1.-ALPHA) * i
                break
            state = next_state
        stats = agent.update(batch)
        if idx % 100 == 0:
            print(f"Iteration {idx:5}: run length: {run_length:8.2f}, critic loss: {stats['loss_val']:8.3f}, actor loss: {stats['loss_pol']:8.3f}, l2 grad: {stats['grad_l2']:8.3f}, ")
            if run_length > params['stop_length']:
                return agent
        
        torch.save(agent.actor, './temp/simple/model/actor.pkl')
        torch.save(agent.critic, './temp/simple/model/critic.pkl')

    return agent

if __name__ == '__main__':
    env = gym.make(params['env_name'])#.unwrapped
    test_env = gym.make(params['env_name'])#.unwrapped
    agent = train(env, test_env)
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.n)
    agent.actor  = torch.load('./temp/simple/model/actor.pkl')
    agent.critic = torch.load('./temp/simple/model/critic.pkl')
    agent.play_episode(env, show=True)
    env.close()