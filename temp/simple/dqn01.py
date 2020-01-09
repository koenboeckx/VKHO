"""
From "PyTorch Deep Learning Hands-On"
"""

from collections import namedtuple
import random
import math
from itertools import count

import gym
import numpy as np 
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward'])

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)
env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
screen_width = 600

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1)) # transpose into torch order (CHW)
    screen = screen[:, 160:320] # strip of bottom of the screen

    # get cart location
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    cart_location = int(env.state[0] * scale + screen_width / 2.) # Middle of cart

    # decide how much to strip
    view_width = 320
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width //2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # strip of the edges, so that we have a square image centererd on a cart
    screen = screen[:, :, slice_range]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    screen = torch.from_numpy(screen)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
    return resize(screen).unsqueeze(0).to(device)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # !!

EPS_START  = 0.9
EPS_END    = 0.05
EPS_DECAY  = 200
steps_done = 0

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    sample = random.random()
    if sample > eps_threshold:
        # freeze network and get predictions
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # select random action
        return torch.tensor([[random.randrange(2)]], 
                device=device, dtype=torch.long)


BATCH_SIZE = 64
GAMMA = 0.999
optimizer = optim.RMSprop(policy_net.parameters())

def optimize_model():
    # don't optimize until at least BATCH_SIZE memories are filled
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # get the actual Q
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    state_values = policy_net(state_batch) # values of states for all actions
    state_action_values = state_values.gather(1, action_batch)

    # get the expected Q
    # mask to identify if next state is final
    non_final_mask = torch.tensor(tuple(map (lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])                                  
    next_state_values = torch.zeros(BATCH_SIZE, device=device) # init to zeros
    # predict next non-final state values from target_net using next states
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    reward_batch = torch.cat(batch.reward)
    # compute the predicted values of states for actions
    expected_state_values = GAMMA * next_state_values + reward_batch

    # compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_values.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1., 1.)
    optimizer.step()

num_episodes  = 20
TARGET_UPDATE = 5

for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for t in count(): # for each timestep in an episode
        # select action for the given state and get rewards
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        # Store transition in memory
        memory.push(state, action, next_state, reward)

        # move to next state
        state = next_state

        # perform one step of the oprimization (on the target network)
        optimize_model()
        if done:
            break

        # update the target network every TARGET_UPDATE episodes
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

env.close()
                            