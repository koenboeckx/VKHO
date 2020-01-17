import gym
import ptan
import numpy as np 
import torch.nn as nn

env = gym.make('CartPole-v0')
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n) 
        )
    def forward(self, x):
        return self.net(x.float())

action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1)
agent = ptan.agent.DQNAgent(Model(), action_selector)

obs = np.array([env.reset()], dtype=np.float32)
print(agent(obs)) # tuple (action_to_take, value related to stateful agents)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
it = iter(exp_source)
print(next(it))