"""
Independent Q-Learning.
"""

import random
import time
import numpy as np
import torch
from torch import nn
import copy

from . import iql_model

from tensorboardX import SummaryWriter

DEBUG_IQL = False
DEBUG_LOSS = True

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
    """
    Agent class to be used with independent q-learning.
    Model and target model are set with method .set_model()
    Syncing from model to target is done with .sync_models()
    Epsilon-greedy action selection with .get_action(state, eps)
    """
    def __init__(self, idx, board_size=11):
        super(IQLAgent, self).__init__()
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
    
    def set_model(self, input_shape, lr, device):
        """
        Set the model (neural net) and target of the agent.

        :param input_shape: shape of input vector (channels, widht, height)
        :param lr:          learning rate for model optimizer (=Adam)
        :param device:      torch.device("cpu" or "cuda")
        :return: None
        """
        self.model  = iql_model.IQL(input_shape, 8,
            lr=lr, board_size=self.board_size).to(device)
        self.target =iql_model.IQL(input_shape, 8,
            lr=lr, board_size=self.board_size).to(device)
        self.sync_models()
    
    def sync_models(self):
        """
        Sync weights of agent.model with agent.target

        :return: None
        """
        self.target.load_state_dict(self.model.state_dict())


    def get_action(self, state, epsilon, device):
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
            state_v = preprocess([state]).to(device)
            values = self.model(state_v)
            return torch.argmax(values).item()

def preprocess(states):
    """
    Process the 'state' such that it can serve
    as input to the NN model.
    
    :param states:  list of states
    :return: tensor 
    """
    size = int(np.sqrt(len(states[0].board)))
    tensor = torch.zeros((len(states), 1, size, size))
    for idx, state in enumerate(states):
        board = state.board
        for i in range(size):
            for j in range(size):
                if board[size*i + j] != -1:
                    tensor[idx, 0, i, j] = int(board[size*i + j][-1]) + 1
    return tensor

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
        assert len(self.contents) >= N
        return random.sample(self.contents, N)

def create_temp_schedule(start, stop, max_steps):
    def get_epsilon(step_idx):
        if step_idx < max_steps:
            return start - step_idx * (start-stop) / max_steps
        else:
            return stop
    return get_epsilon

def calc_loss(agent, batch, gamma, device="cuda"):
    batch = list(zip(*batch))
    states, actions, rewards, next_states, dones = batch
    states_v = preprocess(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.Tensor(rewards).to(device)
    next_v = preprocess(next_states).to(device)
    done_mask = torch.LongTensor(dones).to(device) # 1 if terminated, otherwise 0

    # 6. for every transition, compute y = r is episode ended othrewise y = r + ...
    values_v  = agent.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_values_v = torch.max(agent.target(next_v), dim=1)[0]
    next_values_v[done_mask] = 0.0
    next_values_v = next_values_v.detach() # !! avoids feeding gradients in target network
    targets_v = rewards_v + gamma * next_values_v
    
                    
    # 7. calculate loss L = (Q - y)^2
    loss = nn.MSELoss()(targets_v, values_v)

    return loss
 
def train(env, agents, **kwargs):
    """Train two agents in agent_list"""
    n_steps = kwargs.get('n_steps', 1024)
    mini_batch_size = kwargs.get('mini_batch_size', 32)
    buffer_size = kwargs.get('buffer_size', 128)
    replay_start_size = kwargs.get('replay_start_size', buffer_size)
    gamma = kwargs.get('gamma', 0.99)
    sync_rate = kwargs.get('sync_rate', 10) # when copy model to target?
    print_rate = kwargs.get('print_rate', 100) # print frequency
    save = kwargs.get('save', False) # print frequency
    if torch.cuda.is_available():
        print("CUDA available ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device {} ...'.format(device))
    lr = kwargs.get('lr', 0.02)

    comment = "-lr_{:.3}-batch_{}".format(lr, mini_batch_size)
    with SummaryWriter(comment=comment) as writer:
        
        # create and initialize model for agent
        input_shape = (1, env.board_size, env.board_size)
        for agent in agents:
            agent.set_model(input_shape, lr, device)

        get_epsilon = create_temp_schedule(1.0, 0.1, 50000)

        reward_sum = 0 # keep track of average reward
        n_terminated = 0

        buffers = [ReplayBuffer(buffer_size) for _ in agents]
        state = env.get_init_game_state()
        
        for step_idx in range(int(n_steps)):
            
            eps = get_epsilon(step_idx)
            actions = [0, 0, 0, 0]
            for agent in env.agents:
                if agent in agents:
                    # 2. with prob epsilon, select random action a; otherwise a = argmax Q(s, .)
                    actions[agent.idx] = agent.get_action(state, epsilon=eps, device=device)
                else:
                    actions[agent.idx] = agent.get_action(state)
                    #actions[agent.idx] = 0 # force all other player to stand still
            
            # 3. execute actions, get next state and rewards TODO: get (.., observation, ..) in stead of action
            next_state = env.step(state, actions)
            reward = env.get_reward(next_state)

            #env.render(state)
            done = 0.
            if env.terminal(next_state) != 0:
                print('@ iteration {}: episode terminated; rewards = {}'.format(
                    step_idx, reward))
                n_terminated += 1
                reward_sum += reward[0]
                done = 1.
                next_state =  env.get_init_game_state()

            # 4. store transition (s, a, r, s') in replay buffer
            for idx, agent in enumerate(agents):
                buffers[idx].insert((state, actions[agent.idx],
                                    reward[agent.idx], next_state, # states are common TODO: use observations
                                    done))
        
            if len(buffers[0]) >= replay_start_size: # = minibatch size
                for agent_idx, agent in enumerate(agents):
                    # Sample minibatch and restructure for input to agent.model and loss calculation
                    # 5. sample a random minibatch of transitions from the replay buffer
                    minibatch = buffers[agent_idx].sample(mini_batch_size)
                    loss =calc_loss(agent, minibatch, gamma, device=device)

                    writer.add_scalar('agent{}_loss'.format(agent.idx), loss.item(), step_idx)
                    if DEBUG_LOSS and step_idx % 100 == 0:
                        print('Player {} -> loss = {}'.format(agent_idx, loss.item()))
                
                    # perform training step 
                    # 8. Update Q(s,a) using SGD by minimizing loss w.r.t. model parameters
                    loss.backward()
                    agent.model.optim.step()

                    if step_idx > 0 and step_idx % sync_rate == 0:
                        print('iteration {} - syncing ...'.format(step_idx))
                        agent.sync_models()

                    
            if step_idx > 0 and step_idx % print_rate == 0:
                if n_terminated > 0:
                    print('Iteration {} - Average reward team 0: {} [terminations = {}]'.format(
                            step_idx, reward_sum/n_terminated, n_terminated))
                    writer.add_scalar('win_rate', reward_sum/n_terminated, step_idx)
                else:
                    print('Iteration {} - no terminations'.format(step_idx))
                reward_sum = 0
                n_terminated = 0
            
            state = next_state
        
        if save:
            rand_int = random.randint(1000, 2000)
            for agent in agents:
                filename =  './marl/models/iql_agent_{}_{}.torch'.format(agent.idx, str(rand_int))
                torch.save(agent.model.state_dict(), filename)
        
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