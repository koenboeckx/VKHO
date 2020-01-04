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
    'episodes_to_train':    100,
}

class Model(nn.Module):
    def __init__(self, obs_shape, n_actions, n_hidden=128):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(obs_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.value  = nn.Linear(n_hidden, 1)
        self.policy = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.common(x.float())
        val = self.value(x)
        pol = self.policy(x)
        return pol, val

class Agent:
    def __init__(self, obs_shape, n_actions):
        self.n_actions = n_actions
        self.model = Model(obs_shape, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), params['learning_rate'])
    
    def choose_action(self, observation):
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(observation))
            action = Categorical(logits=logits).sample().item()
        return action
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        self.optimizer.zero_grad()

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)

        logits, curr_vals = self.model(states_v)
        _, next_vals      = self.model(next_states_v)
        next_vals[dones]  = 0.0 # mask away terminal states
        pred_vals = rewards_v.unsqueeze(1) + params['gamma'] * next_vals 
        loss_val = F.mse_loss(curr_vals, pred_vals)

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_actions = log_probs[range(len(batch)), actions_v]
        loss_pol = log_prob_actions * (pred_vals - curr_vals.detach()).squeeze()
        loss_pol = -loss_pol.mean() # used to be .sum()

        loss = loss_val + loss_pol
        loss.backward()
        self.optimizer.step()

        ## compute statistics
        # 1. KL divergence before and after update
        new_logits, _ = self.model(states_v)
        new_probs = F.softmax(new_logits, dim=1)
        kl_div = -((new_probs/probs).log() * probs).sum(dim=1).mean()

        statistics = {
            'loss_pol': loss_pol.item(),
            'loss_val': loss_val.item(),
            'loss':     loss.item(),
            'kl_div':   kl_div.item()
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

def generate_episode(env, agent):
    episode = []
    state, done = env.reset(), False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        episode.append(Experience(state, action, reward, next_state, done))
        state = next_state
    return episode

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
        stats = agent.update(batch)
        print(f"""Episode {episode_idx:4}: policy loss: {stats['loss_pol']:8.4f}, value loss: {stats['loss_val']:8.4f}, running length: {float(running_length):7.3f}""")
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