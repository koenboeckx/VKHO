import random
import logging
from collections import namedtuple
import numpy as np

import pommerman
from pommerman import agents 
from pommerman import characters

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('pommerman')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

logging.basicConfig(filename='./pommerman_test/my_agents.log', level=logging.DEBUG)

class Model00(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
        )
        conv_out_size = self._get_conv_out((1, 11, 11))

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, args.n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _proces_inputs(self, inputs):
        x = torch.zeros((len(inputs), 1, 11, 11))
        for idx, inp in enumerate(inputs):
            x[idx, 0, :, :] = torch.tensor(inp['board'])
        return x
    
    def forward(self, inputs):
        x = self._proces_inputs(inputs)
        x = self.conv(x).view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

class Model02(nn.Module):
    """(Flat) Model that includes other info than pure board.
    Extends model above with additional layer for ammo, alive,
    blast_strength, can_kick."""
    def __init__(self, input_shape, args):
        super().__init__()
        self.in_fc = nn.Sequential( # input fully-connected layer
            nn.Linear(7, 32), # additional input: ammo (4) , alive (1), blast_strength (1), can_kick (1)
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size + 32, 128)
        self.fc2 = nn.Linear(128, args.n_actions)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _proces_inputs(self, inputs):
        x = torch.zeros((len(inputs), 3, 11, 11))
        y = torch.zeros(len(inputs), 7)
        for idx, inp in enumerate(inputs):
            x[idx, 0, :, :] = torch.tensor(inp['board'])
            x[idx, 1, :, :] = torch.tensor(inp['blast_strength'])
            x[idx, 2, :, :] = torch.tensor(inp['bomb_life'])
            for pidx, player in enumerate([11, 12, 13, 14]):
                y[idx, pidx] = 1 if player in inp['alive'] else 0
            y[idx, 4] = torch.tensor(inp['ammo'])
            y[idx, 5] = torch.tensor(inp['blast_strength'])
            y[idx, 6] = torch.tensor(int(inp['can_kick']))
        return x, y
    
    def forward(self, inputs):
        x, y = self._proces_inputs(inputs)
        x = self.conv(x).view(x.size()[0], -1)
        y = self.in_fc(y)
        xy = torch.cat((x, y), 1)
        xy = F.relu(self.fc1(xy))
        pol = self.fc2(xy)
        return pol

class Model03(nn.Module):
    """(Recurrent) Model that includes other info than pure board.
    Extends model above with additional layer for ammo, alive,
    blast_strength, can_kick."""
    def __init__(self, input_shape, args):
        super().__init__()
        # TODO: finish this


class IPGAgent(agents.BaseAgent):
    "Independent Policy Gradients Agent"
    def __init__(self, args, character=characters.Bomber):
        super().__init__(character)
        self.args = args
        self.device = args.device
        self.actor = args.model((3, 11, 11), args).to(args.device)
        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=args.lr)
    
    def act(self, obs, action_space):
        """Choose action from P(action|obs),
        represented by Agent.actor"""

        logits = self.actor([obs])[0]

        #logging.debug(self.actor._proces_inputs([obs]))
        logging.debug('agent {} - obs = {}'.format(self.idx, self.actor._proces_inputs([obs])))
        logging.debug(logits)

        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        print(self.idx, ' -> ', probs.data, 'chosen action = ', action.item(), ' temp = ', self.temperature)
        return action.item()

def generate_episode(env, render=False):
    episode = []
    state = env.reset()
    done = False
    while not done:
        if render: env.render()
        actions = env.act(state)
        next_state, reward, done, info = env.step(actions)
        #print('actions = ', actions, ' alive = ', next_state[0]['alive'])
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

def reinforce(learners, args):
    agent_list = [learners[0],
                  agents.SimpleAgent(),
                  #agents.PlayerAgent(agent_control="arrows"),
                  learners[1],
                  agents.SimpleAgent(),
                  #agents.PlayerAgent(agent_control="wasd")
                ]

    env = pommerman.make('PommeTeamCompetition-v0', agent_list)

    for i_episode in range(args.n_episodes):
        episode = generate_episode(env, render=args.render)
        returns = compute_returns(env, episode, args.gamma)

        ex.log_scalar("episode_length", len(episode))
        for agent in learners:
            ex.log_scalar("reward_{}".format(agent.idx),
                          episode[-1].reward[agent.idx],
                          i_episode)
            print('rewards = ', episode[-1].reward)
        
        episode = list(zip(*episode))
        returns = list(zip(*returns))
        for agent in learners:
            agent.actor.zero_grad()
            
            returns_v = torch.tensor(returns[agent.idx]).to(agent.device)
            states  = [state[agent.idx] for state in episode[0]]
            actions_v = torch.LongTensor([action[agent.idx] for action in episode[1]]).to(agent.device)

            logits_v = agent.actor(states).to(agent.device)
            logprob_v = F.log_softmax(logits_v, dim=1)
            prob_v = F.softmax(logits_v, dim=1)
            
            logprob_act_vals_v = returns_v * logprob_v[range(len(states)), actions_v]
            loss_policy_v = -logprob_act_vals_v.mean()

            entropy_v = - (prob_v * logprob_v).sum(dim=1).mean()
            loss_entropy_v = -args.entropy_beta * entropy_v

            loss_v = loss_policy_v + loss_entropy_v
            
            print('Agent {} - Episode {} -> loss = {}'.format(agent.idx, i_episode, loss_v.item()))

            ex.log_scalar('policy_loss_{}'.format(str(agent.idx)), loss_policy_v.item())
            ex.log_scalar('entropy_loss_{}'.format(str(agent.idx)), loss_entropy_v.item())
            ex.log_scalar('loss_{}'.format(str(agent.idx)), loss_v.item())

            loss_v.backward()
            agent.optimizer.step()

            # TODO: add comparison new and old policy with KL
     
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

Arguments = namedtuple('Arguments', 
                        ['device', 'n_episodes', 'n_actions', 'gamma',
                         'lr', 'entropy_beta', 'model', 'render'])

@ex.config
def cfg():
    agent_type = 'REINFORCE'
    args = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'n_episodes': 100000,
        'n_actions': 6,
        'gamma': 0.99,
        'lr': 0.0001,
        'entropy_beta': 0.01,
        'model': Model00, 
        'render': False
    }

class DictWrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, attr):
        return self.d[attr]

@ex.automain
def run(args):
    args_ = DictWrapper(args)
    # create the two learning agents
    learners = []
    for idx in [0, 2]: # team 0 contains agents 0 and 2
        agent = IPGAgent(args_)
        agent.idx = idx
        learners.append(agent)
    print(learners[0].actor)
    reinforce(learners, args_)
