"""
https://raw.githubusercontent.com/yc930401/Actor-Critic-pytorch/master/Actor-Critic.py
"""

import gym, os
from itertools import count

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torch.distributions import Categorical

DEBUG = False
ALPHA = .9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0').unwrapped

state_size  = env.observation_space.shape[0]
action_size = env.action_space.n 
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size  = state_size
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    
    def forward(self, state):
        output = self.net(state)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size  = state_size
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        value = self.net(state)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(actor, critic, n_iters):
    actor_optimizer  = optim.Adam(actor.parameters())
    critic_optimizer = optim.Adam(critic.parameters())

    run_length = 10.
    for iter in range(n_iters):
        state = env.reset()
        log_probs, values, rewards, masks = [], [], [], []
        entropy = 0.0

        for i in count():
            if DEBUG: env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean() # !!!

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1.-done],  dtype=torch.float, device=device))

            state = next_state

            if done:
                run_length = ALPHA * run_length + (1.-ALPHA) * i
                break

        if iter % 20 == 0:
            print(f"Iteration {iter:4d}: running length = {run_length:8.3f}")
        
        if run_length > 500:
            print(f"Iteration {iter:4d}: running length = {run_length:8.3f}")
            return

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach() # why .detach()? => to prevent changes based on values in Bellman eq to predict
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
    
    env.close()

if __name__ == '__main__':
    actor  = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)
    train(actor, critic, n_iters=1000)
    state, done = env.reset(), False
    while not done:
        env.render()
        state = torch.FloatTensor(state).to(device)
        dist, value = actor(state), critic(state)
        action = dist.sample()
        state, _, done, _ = env.step(action.cpu().numpy())
