"""
Independent Q-Learning.
"""

# TODO: integrate unrolling to longer experience chains

import random
import time
import copy
from datetime import datetime
from collections import namedtuple
#from .common import preprocess
from .common import preprocess_gym_iql as preprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy

from . import agent_models

import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents 

from tensorboardX import SummaryWriter

DEBUG_IQL = False
DEBUG_LOSS = True

Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])



class IQLAgent(agents.BaseAgent):
    """
    Agent class to be used with independent q-learning.
    Model and target model are set with method .set_model()
    Syncing from model to target is done with .sync_models()
    Epsilon-greedy action selection with .get_action(state, eps)
    """
    def __init__(self, idx, device, board_size=11):
        super(IQLAgent, self).__init__()
        self.device = device
        self.board_size = board_size
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = 5000
        self.max_range = 4
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action
    
    def set_model(self, input_shape, n_actions, lr):
        """
        Set the model (neural net) and target of the agent.

        :param input_shape: shape of input vector (channels, widht, height)
        :param lr:          learning rate for model optimizer (=Adam)
        :param device:      torch.device("cpu" or "cuda")
        :return: None
        """
        self.model  = agent_models.IQLModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(self.device)
        self.target = agent_models.IQLModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(self.device)
        self.sync_models()
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.sync_models()
    
    def save_model(self, filename=None):
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = '/home/koen/Programming/VKHO/marl/models/iql_agent{}_{}.torch'.format(
                            self.idx, date_str)
        torch.save(self.model.state_dict(), filename)
    
    def sync_models(self):
        """
        Sync weights of agent.model with agent.target

        :return: None
        """
        self.target.load_state_dict(self.model.state_dict())


    def get_action(self, state, epsilon=0.0):
        """
        Sample an action from action space with epsilon-greedy
        
        :param state:   state of the game
        :param epsilon: value for epsilon-greedy selection
        :param device:  torch.device("cpu" or "cuda")

        :return: action (integer in [0..7])
        """
        if random.random() < epsilon:
            return random.randint(0, 7)
        else:
            state_v = preprocess([state]).to(self.device)
            values = self.model(state_v)
            return torch.argmax(values).item()

class GymAgent:
    def __init__(self, idx, device, **kwargs):
        self.idx = idx
        self.device = device
    
    def __repr__(self):
        return 'gym{}'.format(self.idx)

    def set_model(self, input_shape, n_actions, n_hidden, lr, device):
        self.model = agent_models.GymModelIQL(input_shape, n_actions,
            n_hidden=n_hidden, lr=lr).to(self.device)
        self.target = agent_models.GymModelIQL(input_shape, n_actions,
            n_hidden=n_hidden, lr=lr).to(self.device)
        self.sync_models()

    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_model(self, filename=None):
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = '/home/koen/Programming/VKHO/marl/models/gym_iql_agent{}_{}.torch'.format(
                            self.idx, date_str)
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            state_v = preprocess([state]).to(self.device)
            values = self.model(state_v)
            return torch.argmax(values).item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
    
    def __len__(self):
        return len(self.contents)
    
    def __str__(self):
        return self.contents
    
    def insert(self, item):
        if len(self.contents) == self.capacity:
            del self.contents[0]
        self.contents.append(item)
    
    def sample(self, N):
        if len(self.contents) < N:
            raise ValueError('Length of ReplayBuffer (={}) is smaller than requested # of samples (={})'.format(
                                len(self.contents), N))
        
        return random.sample(self.contents, N)

def create_temp_schedule(start, stop, max_steps):
    def get_epsilon(step_idx):
        if step_idx < max_steps:
            return start - step_idx * (start-stop) / max_steps
        else:
            return stop
    return get_epsilon

def process_batch(batch, device):
    batch = list(zip(*batch))
    states, actions, rewards, next_states, dones = batch
    states_v = preprocess(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.Tensor(rewards).to(device)
    next_v = preprocess(next_states).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    return states_v, actions_v, rewards_v, next_v, done_mask

def calc_loss(agent, batch, gamma):
    """
    Compute MSE loss function on batch according to Q-learning 
    :param agent: instance of IQL agent
    :param batch: list of [..., Experience, ...]
    :param gamma: discount factor
    :param device:

    :return: loss, instance of F.mse_loss(., .)
    """

    states_v, actions_v, rewards_v, next_v, done_mask = process_batch(batch, agent.device)

    # For every transition, compute y = r is episode ended othrewise y = r + ...
    values_v  = agent.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    next_values_v = torch.max(agent.target(next_v), dim=1)[0]
    next_values_v[done_mask] = 0.0
    next_values_v = next_values_v.detach() # !! avoids feeding gradients in target network
    targets_v = rewards_v + gamma * next_values_v
                    
    # Calculate loss L = (Q - y)^2
    loss = F.mse_loss(targets_v, values_v)
    return loss
 
def train(env, agents, **kwargs):
    """Train two agents in agent_list"""
    n_steps = int(kwargs.get('n_steps', 1024))
    mini_batch_size = kwargs.get('mini_batch_size', 32)
    buffer_size = kwargs.get('buffer_size', 128)
    replay_start_size = kwargs.get('replay_start_size', buffer_size)
    gamma = kwargs.get('gamma', 0.99)
    sync_rate = kwargs.get('sync_rate', 10) # when copy model to target?
    ex = kwargs.get('experiment') # instance of sacred.Experiment
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device {} ...'.format(device))
        
    get_epsilon = create_temp_schedule(1.0, 0.1, 500000)

    buffers = [ReplayBuffer(buffer_size) for _ in agents]
    state = env.get_init_game_state()

    env_test = copy.deepcopy(env)

    for step_idx in range(n_steps):        
        eps = get_epsilon(step_idx)
        eps = 0.1 # fixed epsilon
        actions = [0, 0, 0, 0]
        for agent in env.agents:
            if agent in agents:
                # with prob epsilon, select random action a; otherwise a = argmax Q(s, .)
                actions[agent.idx] = agent.get_action(state, epsilon=eps)
            else:
                # other, no trained agent => pick random action
                actions[agent.idx] = agent.get_action(state)
                #actions[agent.idx] = 0 # avoid taking action
        
        # Execute actions, get next state and rewards TODO: get (.., observation, ..) in stead of action
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)

        done = False
        if env.terminal(next_state) != 0:
            done = True
            next_state = env.get_init_game_state()

        # Store transition (s, a, r, s') in replay buffer
        for idx, agent in enumerate(agents):
            exp = Experience(state=state, action=actions[agent.idx],
                             reward=reward[agent.idx],
                             next_state=next_state, done=done)
            buffers[idx].insert(exp)

        if len(buffers[0]) >= replay_start_size: # = minibatch size
            for agent_idx, agent in enumerate(agents):
                agent.model.optimizer.zero_grad()
                # Sample minibatch and compute loss
                minibatch = buffers[agent_idx].sample(mini_batch_size)
                loss = calc_loss(agent, minibatch, gamma)

                ex.log_scalar('agent{}_loss'.format(agent.idx), loss.item())
            
                # perform training step 
                loss.backward()
                agent.model.optimizer.step()

                if step_idx > 0 and step_idx % sync_rate == 0:
                    print('iteration {} - syncing ...'.format(step_idx))
                    agent.sync_models()
                
            # evaluate one episode
            tot_reward = total_reward(play_episode(env_test))
            for agent in env.agents:
                ex.log_scalar('reward_agent{}'.format(agent.idx),
                              tot_reward[agent.idx])

        state = next_state
    
    # Save model when training is over
    for agent in agents:
        agent.save_model()

def play_episode(env):
    episode = []
    state = env.get_init_game_state()
    while not env.terminal(state):
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) else False
        episode.append(Experience(state, actions, reward,
                                    next_state, done))
        state = next_state
    
    return episode

def total_reward(episode):
    cum_reward = episode[0].reward
    for exp in episode[1:]:
        cum_reward = [c + exp.reward[idx] for idx, c in enumerate(cum_reward)]
    return cum_reward


def test(env, agents, filenames=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if filenames is not None:
        # create and initialize model for agent
        input_shape = (1, env.board_size, env.board_size)
        for agent, filename in zip(agents, filenames):
            agent.set_model(input_shape, 0.02, device)
            agent.model.load_state_dict(torch.load(filename))
            agent.model.eval()
    
    state = env.get_init_game_state()

    done = False
    while not done:
        actions = [0, 0, 0, 0]
        for agent in env.agents:
            if agent in agents:
                actions[agent.idx] = agent.get_action(state, epsilon=0.0, # pick best action
                                                        device=device)
            else:
                actions[agent.idx] = agent.get_action(state)
                #actions[agent.idx] = 0 # force all other player to stand still
        
        print(actions)
        next_state = env.step(state, actions)
        print(state.alive)
        reward = env.get_reward(state)
        time.sleep(1.)

        if reward[0] != 0:
            done = True

        env.render(next_state)
        state = next_state

       