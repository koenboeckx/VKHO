import time , sys 

import numpy as np
import torch
from torch import nn 

HYPERPARAMS = {
    'pong': {
        'env_name':             "PongNoFrameskip-v4",
        'stop_reward':          18.0,
        'run_name':             'pong',
        'replay_size':          100000,
        'replay_initial':       10000,
        'target_net_sync':      1000,
        'epsilon_frames':       10**5,
        'epsilon_start':        1.0,
        'epsilon_final':        0.02,
        'learning_rate':        0.0001,
        'gamma':                0.99,
        'batch_size':           32
    },
}

def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state) # this will be masked away
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

def calc_loss_dqn(batch, net, tgt_net, gamma, device='cpu', double=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def calc_values_of_states(states, net, device='cpu'):
    mean_vals = []
    for batch in np.array_split(states, 64): # Split an array into multiple sub-arrays
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)

## Utility classes to simplify training loop
class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)
    
    def frame(self, frame):
        """Update value of epsilon according to standard DQN epsilon decay schedule"""
        self.epsilon_greedy_selector.epsilon = max(self.epsilon_final, self.epsilon_start - frame/self.epsilon_frames)

# to be used as context manager
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
    
    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self
    
    def __exit__(self):
        self.writer.close()
    
    def reward(self, reward, frame, epsilon=None):
        """Computes statistics and outputs them to screen and writer.
        To be called every time an episode finishes"""
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2ff/s%s" % (frame, len(self.total_rewards),
                mean_reward, speed, epsilon_str))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print(f"Solved in {frame} frames")
            return True
        return False