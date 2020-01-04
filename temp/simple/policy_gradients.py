import collections

import numpy as np
import gym
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

params = {
    'learning_rate':        0.001,
    'n_episodes':           20000,
    'gamma':                0.99,
    'stop_length':          400,
    'episodes_to_train':    50,
}

class Model(nn.Module):
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
        return self.net(x.float())

class Agent:
    def __init__(self, obs_shape, n_actions):
        self.n_actions = n_actions
        self.model = Model(obs_shape, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), params['learning_rate'])
    
    def choose_action(self, observation):
        logits = self.model(torch.tensor(observation)).detach()
        #probs  = F.softmax(logits)
        action = Categorical(logits=logits).sample().item()
        return action
    
    def update(self, batch):
        discounted_rewards = discount_batch(batch, params['gamma'])
        states, actions, _, _, dones = zip(*batch)

        self.optimizer.zero_grad()

        states_v = torch.tensor(states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(discounted_rewards)

        logits = self.model(states_v)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_actions = log_probs[range(len(batch)), actions_v]
        loss = log_prob_actions * rewards_v
        loss = -loss.sum()

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def play_episode(self, env, show=False):
        state, done = env.reset(), False
        while not done:
            if show:
                env.render()
            action = self.choose_action(state)
            state, _, done, _ = env.step(action)



Experience = collections.namedtuple('Experience', 
                field_names = ['state', 'action', 'reward', 'next_state', 'done'])

def generate_episode(env, agent):
    episode = []
    state, done = env.reset(), False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        episode.append(Experience(state, action, reward, next_state, done))
        state = next_state
    return episode

def discount_episode(episode, gamma):
    discounted_rewards = []
    R = 0.0
    for exp in reversed(episode):
        R = gamma * R + exp.reward
        discounted_rewards.insert(0, R)
    return discounted_rewards

def discount_batch(batch, gamma):
    discounted_rewards = []
    R = 0.0
    for exp in reversed(batch):
        if exp.done:
            R = 0.0
        R = gamma * R + exp.reward
        discounted_rewards.insert(0, R)
    return discounted_rewards

def train(env):
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.n)
    running_length = None
    batch, batch_episodes = [], 0
    for episode_idx in range(params['n_episodes']):
        episode = generate_episode(env, agent)
        running_length = len(episode) if running_length is None else .9 * running_length + .1 * len(episode)
        batch.extend(episode)
        batch_episodes += 1
        if batch_episodes < params['episodes_to_train']:
            continue
        loss = agent.update(batch)
        print(f'Episode {episode_idx:4}: loss: {loss:8.2f}, current length: {len(episode):3}, running length: {float(running_length):7.3f}')
        if running_length > params['stop_length']:
            print(f'Solved after {episode_idx} episodes')
            break

        batch, batch_episodes = [], 0
    
    return agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    #env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500
    agent = train(env)
    agent.play_episode(env, show=True)
    env.close()