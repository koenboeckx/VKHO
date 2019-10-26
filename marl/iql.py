"""
Independent Q-Learning.
"""

import random
import numpy as np
import torch
from torch import nn
import copy


from . import iql_model

SYNC_RATE = 10 # when copy model to target?
BOARD_SIZE = 11

class BaseAgent:
    """
    This is the base abstraction for agents. 
    All agents should inherit from this class
    """

    def __init__(self):
        pass
    
    def get_action(self, obs, action_space):
        """Return action to be executed by environment"""
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent
        know the episode has ended and what the reward is
        
        Args:
            reward: the single reward scalar to this agent.
        """
        pass

    def __repr__(self):
        return self.type + str(self.idx)

class IQLAgent(BaseAgent):
    def __init__(self, idx):
        super(IQLAgent, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = 5
        self.max_range = 4
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action
    
    def set_model(self, input_shape):
        self.model  = iql_model.IQL(input_shape, 8) # TODO: implement target network
        self.target = copy.deepcopy(self.model)
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())


    def get_action(self, state, epsilon):
        values = self.model(preprocess(state))
        if random.random() < epsilon:
            return random.sample(range(8), 1)[0]
        else:
            return torch.argmax(values).item()

def preprocess(state):
    """process the 'state' such thzt it can serve
    as input to the NN model."""
    board = state.board
    size = int(np.sqrt(len(board)))
    result = np.zeros((1, size, size))
    for i in range(size):
        for j in range(size):
            if board[size*i + j] != -1:
                result[0, i, j] = int(board[size*i + j][-1]) + 1
    return torch.from_numpy(result).type('torch.FloatTensor').unsqueeze(0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
    
    def __len__(self):
        return len(self.contents)
    
    def insert(self, item):
        if len(self.contents) == self.capacity:
            del self.contents[0]
        self.contents.append(item)
    
    def sample(self, N):
        assert len(self.contents) >= N
        return random.sample(self.contents, N)

def create_temp_schedule(start, stop, max_steps):
    def get_epsilon(step_idx):
        if step_idx < max_steps:
            return start - step_idx * (start-stop) / max_steps
        else:
            return stop
    return get_epsilon
 
def train(env, agent, n_steps=20, epsilon=0.1, mini_batch_size=5, buffer_size=10,
            gamma=0.9):
    """Train two agents in agent_list"""
    # create and initialize model for agent
    input_shape = (1, env.board_size, env.board_size)
    agent.set_model(input_shape)

    get_epsilon = create_temp_schedule(1.0, 0.05, 1000)

    buffer = ReplayBuffer(buffer_size)
    state = env.set_init_game_state()
    for step_idx in range(int(n_steps)):
        eps = get_epsilon(step_idx)
        print('epsilon = ', eps)
        action = agent.get_action(state[0], epsilon=eps)
        actions = [0, 0, 0, 0]
        for agent_ in env.agents:
            if agent_ == agent:
                actions[agent.idx] = action
            else:
                actions[agent_.idx] = agent_.get_action(state)
        next_state = env.step(actions)
        reward = env.get_reward()

        print('actions = ', actions)
        env.render()
        if env.terminal():
           next_state =  env.set_init_game_state()
        print('Alive = ', [o.alive for o in state])
        buffer.insert((state[agent.idx], action, reward[agent.idx], next_state[agent.idx]))
    
        if len(buffer) > mini_batch_size: # = minibatch size
            loss = torch.Tensor([0])
            minibatch = buffer.sample(mini_batch_size)
            # TODO: transform part below to do update on total
            states_v  = torch.zeros((len(minibatch), 1, BOARD_SIZE, BOARD_SIZE))
            next_v    = torch.zeros((len(minibatch), 1, BOARD_SIZE, BOARD_SIZE))
            actions_v = torch.LongTensor(np.zeros(len(minibatch))) # one-hot or not?
            rewards_v = torch.zeros(len(minibatch))
            dones_v   = torch.zeros(len(minibatch))
            for idx, (s, a, r, next_s) in enumerate(minibatch):
                states_v[idx, 0, :, :] = preprocess(s)
                next_v[idx, 0, :, :]   = preprocess(next_s)
                actions_v[idx] = int(a)
                rewards_v[idx] = r
                dones_v[idx] = abs(r)
            next_q_v = agent.target(next_v)
            targets_v = rewards_v + (1-dones_v) * torch.max(agent.target(next_v), dim=1)[0]
            values_v  = agent.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            loss = nn.MSELoss()(targets_v, values_v)
            
            # perform training step 
            loss.backward()
            agent.model.optim.step()

        if step_idx > 0 and step_idx % SYNC_RATE == 0:
            agent.sync_models()
