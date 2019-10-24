"""
Independent Q-Learning.
"""

import random
import numpy as np
import torch


from . import iql_model

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
        self.model = iql_model.IQL(input_shape, 8)

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
 
def train(env, agent, n_steps=20, epsilon=1.0, mini_batch_size=5, gamma=0.9):
    """Train the first agent in agent_list"""
    input_shape = (1, env.board_size, env.board_size)
    agent.set_model(input_shape)
    buffer = ReplayBuffer(10)
    state = env.set_init_game_state()
    for _ in range(n_steps):
        action = agent.get_action(state[0], epsilon=0.5)
        actions = (action, )
        for other_agent in env.agents[1:]:
            actions += (other_agent.get_action(state),)
        next_state = env.step(actions)
        reward = env.get_reward()
        buffer.insert((state[0], action, reward[0], next_state[0]))
    
        if len(buffer) > mini_batch_size: # = minibatch size
            loss = torch.Tensor([0])
            minibatch = buffer.sample(mini_batch_size)
            for s, a, r, next_s in minibatch:
                if r != 0: # next_s is terminal
                    y = r
                else:
                    y = r + gamma * torch.max(agent.model(preprocess(next_s))).item()
                loss += (agent.model(preprocess(next_s))[0][a] - y)**2
            
            # perform training step 
            loss.backward()
            agent.model.optim.step()




