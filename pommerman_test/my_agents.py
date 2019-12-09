import random
from collections import namedtuple
import numpy as np

import pommerman
from pommerman import agents as pommerman_agents
from pommerman import characters

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('pommerman')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

class Model01(nn.Module):
    """Simple flat model with only the different 'boards'
    ('board', 'blast_strength', 'bomb_life') as input-
    channels for a convolutional layer.
    Returns tensor of size args.n_actions
    """
    def __init__(self, input_shape, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, args.n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _proces_inputs(self, inputs):
        x = torch.zeros((len(inputs), 3, 11, 11))
        for idx, inp in enumerate(inputs):
            x[idx, 0, :, :] = torch.tensor(inp['board'])
            x[idx, 1, :, :] = torch.tensor(inp['blast_strength'])
            x[idx, 2, :, :] = torch.tensor(inp['bomb_life'])
        return x

    
    def forward(self, inputs):
        x = self._proces_inputs(inputs)
        x = self.conv(x).view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class IPGAgent(pommerman_agents.BaseAgent):
    "Independent Policy Gradients Agent"
    def __init__(self, args, character=characters.Bomber):
        super().__init__(character)
        self.args = args
        self.device = args.device
        self.actor = Model01((3, 11, 11), args).to(args.device)
        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=args.lr)
    
    def act(self, obs, action_space):
        """Choose action from P(action|obs),
        represented by Agent.actor"""
        logits = self.actor([obs])[0]
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()

def generate_episode(env, render=False):
    episode = []
    state = env.reset()
    done = False
    while not done:
        if render: env.render()
        actions = env.act(state)
        next_state, reward, done, info = env.step(actions)
        episode.append(Experience(state, actions, reward, next_state, done))
        state = next_state
    return episode

def compute_returns(env, episode, gamma):
    returns = []
    cum_reward = [0.0,] * len(env._agents)
    for _, _, reward, _, _ in reversed(episode):
        cum_reward = [gamma * c + r for c, r in zip(cum_reward, reward)]
        returns.append(cum_reward)
    return list(reversed(returns))

def reinforce(agents, args):
    agent = agents[0]
    agent_list = [agent,
                  pommerman_agents.SimpleAgent()]
    
    agent.idx = 0
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(args.n_episodes):
        episode = generate_episode(env)
        returns = compute_returns(env, episode, args.gamma)

        ex.log_scalar("reward", episode[-1].reward[agent.idx])
        
        episode = list(zip(*episode))
        returns = list(zip(*returns))
        for agent in agents:
            agent.actor.zero_grad()
            returns_v = torch.tensor(returns[agent.idx]).to(agent.device)
            states  = [state[agent.idx] for state in episode[0]]
            actions_v = torch.LongTensor([action[agent.idx] for action in episode[1]]).to(agent.device)

            logits_v = agent.actor(states)
            logprob_v = F.log_softmax(logits_v, dim=1)
            logprob_act_vals_v = returns_v * logprob_v[range(len(states)), actions_v]

            loss = - logprob_act_vals_v.mean()
            args.ex.log_scalar('loss_{}'.format(str(agent.idx)), loss.item())

            loss.backward()
            agent.optimizer.step()

        
    env.close()

def main(agent):
    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agent
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            print(reward)
        print('Episode {} finished'.format(i_episode))
    env.close()

class Arguments:
    agent_type = 'REINFORCE'
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_episodes = 100000
    n_actions = 6
    gamma = 0.9
    lr = 0.01

@ex.config
def cfg():
    args = Arguments()

@ex.automain
def run(args):
    print(args.agent_type)
    args.ex = ex
    agent = IPGAgent(args)
    print(agent.actor)
    reinforce([agent], args)
