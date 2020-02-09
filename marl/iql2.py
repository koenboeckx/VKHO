import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

import random
import argparse
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 

from envs import Environment, all_actions

STORE = True
DEBUG = False
RENDER = False

if STORE:
    from sacred import Experiment
    from sacred.observers import MongoObserver
    ex = Experiment('IQL_OBS')
    ex.observers.append(MongoObserver(url='localhost',
                                    db_name='my_database'))

params = {
    'n_steps':              5000,
    'board_size':           5,
    'gamma':                0.99,
    'learning_rate':        0.0005, # from pymarl: 0.0005
    'step_penalty':         0.01,
    'buffer_size':          1024,
    'batch_size':           16,
    'max_grad':             10, # Reduce magnitude of gradients above this L2 norm
    'sync_interval':        100, # !!
    'max_episode_length':   200, # limits the play_out of an episode
    'final_epsilon':        0.1,
    'print_interval':       20,
}

agent_params = {
    'init_ammo':            5000,
    'view_size':            7,
    'max_range':            7,
}

if STORE:
    @ex.config
    def cfg():
        params = params
        agent_params = agent_params

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

class IQLModel(nn.Module):
    def __init__(self, input_shape, n_actions, n_hidden=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(self.conv_out_size, n_hidden)

        self.value  = nn.Linear(n_hidden, n_actions)
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float()
        x = self.conv(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))
        value  = self.value(x)
        return value

#----------------------- Agents -----------------------------------------
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
    
    def get_action(self, obs, state):
        return random.randint(0, 7)

class StaticTank(Tank):
    """Tank that does nothing (always 'do_nothing')"""
    def __init__(self, idx, team="friend"):
        super(StaticTank, self).__init__(idx, team)
    
    def get_action(self, obs):
        return 0

class IQLAgent(Tank):
    def __init__(self, idx, models, device, team="friend"):
        super(IQLAgent, self).__init__(idx, team)
        self.device = device
        self.n_actions = len(all_actions)
        self.model = models["model"]
        self.target_model = models["target"]
        self.epsilon_scheduler = LinearScheduler(start=1.0, 
                                                 finish=params['final_epsilon'],
                                                 length=int(1e5))
    
    def sync_models(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, observation, state):
        epsilon = self.epsilon_scheduler()
        if random.random() < epsilon:
            return np.random.choice(range(self.n_actions))
        else:
            with torch.no_grad(): # TODO: check if needed => [02Feb20] no improvement
                q_vals = self.model(torch.tensor([observation])).squeeze()
                # remove actions that are not allowed
                unavailable_actions = self.get_unavailable_actions(state) # TODO: avoid using state; use observation
                q_vals[unavailable_actions] = -np.infty
                action = q_vals.max(0)[1].item()
                return action
    
    def get_unavailable_actions(self, state):
        return self.env.get_unavailable_actions(state, self)

    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = [action[self.idx] for action in actions] # only keep own actions
        rewards = [reward[self.idx] for reward in rewards] # only keep own reward
        rewards_v = torch.tensor(rewards)
        dones_v   = torch.tensor(dones, dtype=torch.float)
        observations_v = torch.tensor([self.env.get_observation(state, self) for state in states])
        observations_next_v = torch.tensor([self.env.get_observation(state, self) for state in next_states])
        
        qvals = self.model(observations_v)
        qvals_chosen_actions = qvals[range(len(batch)), actions]

        target_qvals = self.target_model(observations_next_v)
        unavailable_actions = [self.get_unavailable_actions(state)
                                for state in next_states]
        for idx, unavail in enumerate(unavailable_actions): # eliminate actions that are not allowed
            target_qvals[idx, unavail] = -np.infty
        target_max_qvals = target_qvals.max(1)[0]

        targets = rewards_v + params['gamma'] * (1.-dones_v) * target_max_qvals

        td_error = (qvals_chosen_actions - targets.detach())
        
        loss = (td_error**2).mean()
        self.model.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), 
                                                    params['max_grad'])
        self.model.optimizer.step()

        stats = {
            'loss':     loss.item(),
        }
        return stats



# -----------------------------------  schedulers--------------------------------------

class LinearScheduler:
    def __init__(self, start, finish, length):
        self.start  = start
        self.finish = finish
        self.delta  = (start - finish) / length
        self.t = 0
    
    def __call__(self):
        epsilon = max(self.finish, self.start - self.t * self.delta)
        self.t += 1
        return epsilon

# -----------------------------------  helpers --------------------------------------

class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def __len__(self):
        return len(self.buffer)
    
    def insert(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def insert_batch(self, batch):
        for exp in batch:
            self.insert(exp)

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        return random.sample(self.buffer, batch_size)

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

def play_episode(env, render=False):
    episode = []
    state = env.get_init_game_state()
    step = 0
    #while True:
    for _ in range(params['max_episode_length']):
        observations = env.get_all_obs(state)
        actions = [agent.get_action(obs, state) for agent, obs in zip(env.agents, observations)]
        if render:
            print(f'Step {step}')
            env.render(state)
            print([all_actions[a] for a in actions])
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) != 0 else False
        #done = done or state.ammo[0] == 0 # !! extra termination rule for 1 v 3 static agents
        episode.append(Experience(state, actions, reward, next_state, done))
        if done:
            if render: print(f'Episode terminated after {step} steps with reward = {reward[0]}')
            return episode
        state = next_state
        step += 1
    if render: print(f'Maximum # of steps exceeded')
    return episode

# -----------------------------------  main --------------------------------------

def train(env, learners, others):
    n_wins = 0
    replay_buffer = ReplayBuffer(buffer_size=params['buffer_size'])
    for idx in range(params['n_steps']):
        episode = play_episode(env)
        cum_reward = sum([exp.reward[0] for exp in episode])
        if episode[-1].reward[0] == 1:
            win = 1 
        elif episode[-1].reward[0] == -1:
            win = -1
        n_wins += win
        replay_buffer.insert_batch(episode)
        if len(replay_buffer) < params['batch_size']:
            continue
        batch = replay_buffer.sample(params['batch_size'])
        for agent in learners:
            stats = agent.update(batch)
            if idx > 0 and idx % params['sync_interval'] == 0:
                agent.sync_models()

            if STORE:
                ex.log_scalar(f'loss{agent}', stats['loss'], step=idx)
                ex.log_scalar(f'epsilon{agent}', agent.epsilon_scheduler(), step=idx)
            
        if idx % params['print_interval'] == 0:
            print(f"Step {idx}: Average win rate: {n_wins/params['print_interval']}")
            n_wins = 0
            if RENDER: _ = play_episode(env, render=True)
        if STORE:
            ex.log_scalar(f'episode length', len(episode), step=idx)
            ex.log_scalar(f'win', win, step=idx)
            ex.log_scalar(f'episode_reward', cum_reward, step=idx)

def create_model(n_actions=8):
    view_size = agent_params['view_size']
    input_shape = (4, 2*view_size+1, 2*view_size+1)
    model = IQLModel(input_shape, n_actions)
    model.optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    target = IQLModel(input_shape, n_actions)
    target.load_state_dict(model.state_dict())
    return {'model': model,
            'target': target}


@ex.automain
def run(params, agent_params):
    print(params)
    with open(__file__) as f: # print own source code -> easier follow-up in sacred / mongodb
        print(f.read()) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = create_model()
    learners = [IQLAgent(idx, models, device, team="friend") for idx in [0, 1]]
    others   = [RandomTank(idx, team="enemy") for idx in [2, 3]] # turn off extra stop criterion if not StaticTank
    agents = sorted(learners + others, key=lambda x: x.idx)

    env = Environment(agents, size=params['board_size'],
                        step_penality=params['step_penalty'])

    train(env, learners, others)
    """
    for agent in learners:
        agent.save(f'agent{agent.idx}-iql.pkl')
    """

if __name__ == "__main__" and not STORE:
    run(params)