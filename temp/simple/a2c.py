"""
TODO: compare with https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py
      especially: computing advantage (with compute_returns) -> requires rewrite to episode based 
"""      

import collections, random

import numpy as np
import gym
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

params = {
    'learning_rate':        0.0001,
    'n_steps':              10000,
    'buffer_size':          500,
    'batch_size':           10,
    'gamma':                0.99,
    'stop_length':          400,
    'episodes_to_train':    100,
    'env_name':             'CartPole-v1',
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
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)

        # value loss
        logits    = self.actor(states_v)
        curr_vals = self.critic(states_v)
        next_vals = self.critic(next_states_v)
        next_vals[dones]  = 0.0 # mask away terminal states
        pred_vals = rewards_v.unsqueeze(1) + params['gamma'] * next_vals 
        loss_val = F.mse_loss(curr_vals, pred_vals)

        # policy loss
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_actions = log_probs[range(len(batch)), actions_v]
        loss_pol = log_prob_actions * (pred_vals.detach() - curr_vals.detach()).squeeze()
        loss_pol = -loss_pol.sum() # used to be .sum() (sometimes .mean())

        # entropy loss
        # loss_entropy = -(probs * log_probs).sum(dim=1).mean()

        #loss = loss_val + loss_pol + params['entropy_coeff'] * loss_entropy
        #loss.backward()
        loss_val.backward()
        self.critic_optimizer.step()
        
        loss_pol.backward()
        self.actor_optimizer.step()

        ## compute statistics
        # 2. gradients
        grad_max, grad_means, grad_counts = 0.0, 0.0, 0
        for p in self.actor.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means  += (p.grad ** 2).mean().sqrt().item()
            grad_counts += 1

        statistics = {
            'loss_pol':     loss_pol.item(),
            'loss_val':     loss_val.item(),
            #'loss':         loss.item(),
            #'kl_div':       kl_div.item(),
            'grad_max':     grad_max,
            'grad_l2':      grad_means/grad_counts,
            #'entropy':      loss_entropy.item(),
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
    running_length = None
    buffer = Buffer(size=params['buffer_size'])
    exp_source = Source(env, agent)
    episode_length = 0
    for idx in range(params['n_steps']):
        buffer.insert(exp_source.get_steps(10))
        if len(buffer) < params['batch_size']:
            continue
        batch = buffer.sample(params['batch_size'])
        stats = agent.update(batch)

        if idx > 0 and idx % params['eval_rate'] == 0:
            length = eval(agent, test_env)
            running_length = length if running_length is None else .9 * running_length + .1 * length
            print(f"""Step {idx:4}: policy loss: {stats['loss_pol']:8.4f}, value loss: {stats['loss_val']:8.4f}, grad_l2: {stats['grad_l2']:5.3f}, running length: {float(running_length):7.3f}""")
            if running_length > params['stop_length']:
                print(f'Solved after {idx} steps')
                break
    return agent

if __name__ == '__main__':
    env = gym.make(params['env_name'])
    test_env = gym.make(params['env_name'])
    #env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500
    agent = train(env, test_env)
    agent.play_episode(env, show=True)
    env.close()