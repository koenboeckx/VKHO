"""
[07Feb20] New version to work with observations
"""

import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

from envs import Environment, all_actions
from collections import namedtuple
import random, time, pickle

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])
ALPHA = 0.99 # used to compute running reward
DEBUG = False
STORE = True # set to true to store results with sacred

if STORE:
    from sacred import Experiment
    from sacred.observers import MongoObserver
    ex = Experiment('PG3_OBS')
    ex.observers.append(MongoObserver(url='localhost',
                                    db_name='my_database'))

params = {
    'n_steps':              500,
    'board_size':           7,
    'gamma':                0.99,
    'learning_rate':        0.001,
    'entropy_beta':         0.01,
    'n_episodes_per_step':  40, # 20
    'step_penalty':         0.05, # to induce shorter episodes
    'gru':                  True,
    'type':                 'reinforce', # 'reinforce' or 'a2c' of 'reinforce baseline'
}

agent_params = {
    'init_ammo':            5,
    'view_size':            7,
    'max_range':            5,
}

@ex.config
def cfg():
    params = params
    agent_params = agent_params

# ----------------------- Models -----------------------------------------

class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(self.conv_out_size, 128)

        self.policy = nn.Linear(128, n_actions)
        self.value  = nn.Linear(128, 1)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float()
        x = self.conv(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.policy(x)
        value  = self.value(x)
        return value, logits

# ----------------------- Agents -----------------------------------------
class Tank:
    def __init__(self, idx, team="friend"):
        super(Tank, self).__init__()
        self.init_agent(idx)
        self.team = team
        
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.init_ammo = agent_params['init_ammo']
        self.ammo = agent_params['init_ammo']
        self.max_range = agent_params['max_range']
        self.obs_space = agent_params['view_size']
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action
        
    
    def __repr__(self):
        return self.type + str(self.idx)
    
    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)

class RandomTank(Tank):
    def __init__(self, idx, team="friend"):
        super(RandomTank, self).__init__(idx, team)
    
    def get_action(self, obs):
        return random.randint(0, 7)

class StaticTank(Tank):
    """Tank that does nothing (always 'do_nothing')"""
    def __init__(self, idx, team="friend"):
        super(StaticTank, self).__init__(idx, team)
    
    def get_action(self, obs):
        return 0

class A2CAgent(Tank):
    def __init__(self, idx, model, device, team="friend"):
        super().__init__(idx, team)
        self.model = model
        self.device = device
    
    def get_action(self, observation):
        with torch.no_grad():
            _, logits = self.model(torch.tensor([observation]))
            probs  = F.softmax(logits, dim=-1)
            action = Categorical(probs).sample()
        return action.item()
    
    def discount_rewards(self, batch):
        _, _, rewards, _, dones = zip(*batch)
        returns, R = [], 0.0
        own_rewards = [reward[self.idx] for reward in rewards]
        for reward, done in reversed(list(zip(own_rewards, dones))):
            if done: 
                R = 0.0
            R = reward + params['gamma'] * R
            returns.insert(0, R)
        return returns

    def update_pg(self, batch):
        self.model.optimizer.zero_grad()

        states, actions, _, _, _ = zip(*batch)
        observations_v = torch.tensor([self.env.get_observation(state, self) for state in states])
        
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.LongTensor(own_actions)

        returns = self.discount_rewards(batch)
        returns_v = torch.tensor(returns)

        _, logits_v = self.model(observations_v)
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        #logprob_act_vals_v = returns_v * logprob_actions_v
        logprob_act_vals_v = (returns_v - returns_v.mean())* logprob_actions_v
        loss = -logprob_act_vals_v.mean()

        loss.backward()
        
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
            'grads_var':    np.var(grads),
        }
        self.model.optimizer.step()
        return stats

    def update(self, batch):
        if params['type'] == 'reinforce': # or 'a2c' of 'reinforce baseline'
            return self.update_pg(batch)
        elif params['type'] == 'reinforce baseline':
            return self.update_pg_baseline(batch)
        elif params['type'] == 'a2c':
            return self.update_a2c(batch)


    def _create_stats(self, loss, policy_loss, value_loss, entropy):
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'policy_loss':  policy_loss.item(),
            'value_loss':   value_loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
            'grads_var':    np.var(grads),
            'entropy':      entropy.item(),
        }
        return stats 

def preprocess(states):
    """Process state to serve as input to convolutionel net."""
    bs = params['board_size']
    boards = np.zeros((len(states), 1, bs, bs))
    other  = np.zeros((len(states), 1, bs, 8))
    for idx, state in enumerate(states):
        board = np.array([int(b) for b in state.board])
        board = np.reshape(board, (1, bs, bs))
        boards[idx] = board
        other[idx, 0, 0] = state.alive + tuple(ammo/agent_params['init_ammo'] for ammo in state.ammo)
    return torch.tensor(np.concatenate((boards, other), axis=-1))

def play_episode(env, agents, render=False):
    episode = []
    state = env.get_init_game_state()
    while True:
        #actions = env.get_actions(state) # !! avoids dead agents take action -> gives wrong idea to agent (or) disuades exploration?
        observations = env.get_all_obs(state)
        actions = [agent.get_action(obs) for agent, obs in zip(agents, observations)]
        if render:
            env.render(state)
            print(f"Actions = {[all_actions[a] for a in actions]}")
            time.sleep(1)
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) != 0 else False
        #done = done or state.ammo[0] == 0 # !! extra termination rule for 1 v 3 static agents
        episode.append(Experience(state, actions, reward, next_state, done))
        if done:
            return episode
        state = next_state

def train(env, learners, others):
    stats, grads = {}, {}
    for agent in learners:
        grads[agent.idx] = []
    agents = sorted(learners + others, key=lambda x: x.idx)
    for idx in range(params['n_steps']):
        batch = []
        total_reward = 0
        for epi_idx in range(params['n_episodes_per_step']):
            render = True if DEBUG and epi_idx % 10 == 0 else False
            episode = play_episode(env, agents, render=render)
            reward  = episode[-1].reward[0] 
            total_reward += reward
            batch.extend(episode)

        for agent in learners:
            stats[agent.idx] = agent.update(batch)
            grads[agent.idx].append(stats[agent.idx]['grads_l2'])

        if STORE:
            ex.log_scalar('win_rate', total_reward/ params['n_episodes_per_step'], step=idx)
            ex.log_scalar('mean_length', len(batch) / params['n_episodes_per_step'], step=idx)
            process_stats(idx, stats)
        else:
            print(f"{idx:5d}: win_rate = {total_reward/ params['n_episodes_per_step']:08.7f} - mean length = {len(batch) / params['n_episodes_per_step']:6.2f}")

# ------- helper functions -----------------

def process_stats(idx, stats):
    for agent_idx in stats:
        ex.log_scalar(f'loss{agent_idx}', stats[agent_idx]['loss'], step=idx)
        #ex.log_scalar(f'policy_loss{agent_idx}', stats[agent_idx]['policy_loss'], step=idx)
        #ex.log_scalar(f'value_loss{agent_idx}', stats[agent_idx]['value_loss'], step=idx)
        ex.log_scalar(f'grad{agent_idx}', stats[agent_idx]['grads_l2'], step=idx)
        #ex.log_scalar(f'entropy{agent_idx}', stats[agent_idx]['entropy'], step=idx)
        ex.log_scalar(f'grad_var{agent_idx}', stats[agent_idx]['grads_var'], step=idx)

def create_model():
    view_size = agent_params['view_size']
    input_shape = (4, 2*view_size+1, 2*view_size+1)
    model = A2CModel(input_shape, n_actions=8)
    model.optimizer = optim.Adam(model.parameters(),
                                lr=params['learning_rate'])
    return model

@ex.automain
def run(params, agent_params):
    print(params)
    with open(__file__) as f: # print own source code -> easier follow-up in sacred / mongodb
        print(f.read()) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model()

    learners = [A2CAgent(idx, model, device, team="friend") for idx in [0, 1]]
    others   = [RandomTank(idx, team="enemy") for idx in [2, 3]] # turn off extra stop criterion if not StaticTank
    agents = sorted(learners + others, key=lambda x: x.idx)

    env = Environment(agents, size=params['board_size'],
                        step_penality=params['step_penalty'])
    train(env, learners, others)
    for agent in learners:
        agent.save(f'agent{agent.idx}-iql.pkl')

if __name__ == "__main__" and not STORE:
    run(params)
