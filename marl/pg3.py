"""
[29Jan20] Implement COMA
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
STORE = False # set to true to store results with sacred

if STORE:
    from sacred import Experiment
    from sacred.observers import MongoObserver
    ex = Experiment('new_pg')
    ex.observers.append(MongoObserver(url='localhost',
                                    db_name='my_database'))

params = {
    'n_steps':              200,
    'board_size':           7,
    'gamma':                0.99,
    'learning_rate':        0.001,
    'entropy_beta':         0.01,
    'n_episodes_per_step':  40, # 20
    'init_ammo':            5,
    'step_penalty':         0.05, # to induce shorter episodes
    'gru':                  False,
    'type':                 'a2c', # 'reinforce' or 'a2c' of 'reinforce baseline'
}

#@ex.config
def cfg():
    params = params

# ----------------------- Models -----------------------------------------

class CommonModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = self._get_conv_out(input_shape)
        self.full_in = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.common = nn.Sequential(
            nn.Linear(self.conv_out_size + 64, 128),
            nn.ReLU(),
        )
    
    def _get_conv_out(self, shape):
        """returns the size for fully-connected layer, 
        after passage through convolutional layer"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Input x has two parts: a 'board' for the conv layer,
        and an 'other', containing ammo and alive flags for
        the fully connecte layer."""
        bs = params['board_size']
        x = x.float()
        board = x[:, :, :, :bs]
        other = x[:, :, 0, bs:bs + 8].view(x.size()[0], -1)
        conv_out = self.conv(board).view(x.size()[0], -1)
        full_out = self.full_in(other)
        common_in  = torch.cat((conv_out, full_out), 1)
        common_out = self.common(common_in)
        return common_out

class Actor(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.common = CommonModel(input_shape)
        self.policy = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=params['learning_rate'])
    
    def forward(self, x):
        common_out = self.common(x)
        logits = self.policy(common_out)
        return logits

class Critic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.common = CommonModel(input_shape)
        self.values = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=params['learning_rate'])
    
    def forward(self, x):
        common_out = self.common(x)
        values = self.values(common_out)
        return values
##----------------------- Agents -----------------------------------------
class Tank:
    def __init__(self, idx):
        super(Tank, self).__init__()
        self.init_agent(idx)
    
    def init_agent(self, idx):
        self.type = 'T'
        self.idx  = idx

        # specific parameters
        self.alive = 1
        self.ammo = params['init_ammo']
        self.max_range = 5
        self.pos = None     # initialized by environment
        self.aim = None     # set by aim action 
    
    def __repr__(self):
        return self.type + str(self.idx)
    
    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)

class RandomTank(Tank):
    def __init__(self, idx):
        super(RandomTank, self).__init__(idx)
    
    def get_action(self, obs):
        return random.randint(0, 7)

class StaticTank(Tank):
    """Tank that does nothing (always 'do_nothing')"""
    def __init__(self, idx):
        super(StaticTank, self).__init__(idx)
    
    def get_action(self, obs):
        return 0

class A2CAgent(Tank):
    def __init__(self, idx, device):
        super().__init__(idx)
        self.device = device
        input_shape = (1, params['board_size'], params['board_size'])
        self.actor  = Actor(input_shape, n_actions=8)
        self.critic = Critic(input_shape, n_actions=8)
        
    
    def get_action(self, state):
        with torch.no_grad():
            logits = self.actor(preprocess([state]))
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

    def _compute_returns(self, next_values, rewards, done_mask):
        pass

    def update_pg(self, batch):
        self.actor.optimizer.zero_grad()

        states, actions, _, _, _ = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.LongTensor(own_actions)

        returns = self.discount_rewards(batch)
        returns_v = torch.tensor(returns)

        logits_v = self.actor(preprocess(states))
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        #logprob_act_vals_v = returns_v * logprob_actions_v
        logprob_act_vals_v = (returns_v - returns_v.mean())* logprob_actions_v
        loss = -logprob_act_vals_v.mean()

        loss.backward()
        
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.actor.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
            'grads_var':    np.var(grads),
        }
        self.actor.optimizer.step()
        return stats
    
    def update_critic(self, episode):
        self.critic.zero_grad()
        states, actions, rewards, next_states, dones = zip(*episode)
        own_actions = [action[self.idx] for action in actions]
        own_rewards = [reward[self.idx] for reward in rewards]
        next_q = self.critic(preprocess(next_states)).max(1)[0]
        y = torch.tensor(own_rewards) + params['gamma'] * next_q
        q = self.critic(preprocess(states))[range(len(episode)), own_actions]
        loss = F.mse_loss(y, q)
        loss.backward()
        self.critic.optimizer.step()
        return loss
    
    def update_actor(self, batch):
        self.actor.zero_grad()

        states, actions, rewards, next_states, dones = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        own_rewards = [reward[self.idx] for reward in rewards]
        actions_v = torch.LongTensor(own_actions)
        rewards_v = torch.Tensor(own_rewards)
        dones_v   = torch.FloatTensor(dones)

        logits_v = self.actor(preprocess(states))
        values_v = self.critic(preprocess(states))[range(len(batch)), own_actions]
        
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = values_v.detach() * logprob_actions_v
        loss = -logprob_act_vals_v.mean()

        probs_v = F.softmax(logits_v, dim=1).detach()
        entropy = -((probs_v * logprobs_v.detach()).sum(dim=1).mean())

        loss.backward()
        self.actor.optimizer.step()
        return loss, entropy

    def update_a2c(self, batch):
        # split batch in episodes and done one critic update per episode
        episode = []
        for exp in batch:
            episode.append(exp)
            if exp.done: # perform critic update
                _ = self.update_critic(episode)
                episode = []
        
        loss, entropy = self.update_actor(batch)

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.actor.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
            'grads_var':    np.var(grads),
            'entropy':      entropy.item()
        }
        return stats


    def update(self, batch):
        if params['type'] == 'reinforce': # or 'a2c' of 'reinforce baseline'
            return self.update_pg(batch)
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
        other[idx, 0, 0] = state.alive + tuple(ammo/params['init_ammo'] for ammo in state.ammo)
    return torch.tensor(np.concatenate((boards, other), axis=-1))

def play_episode(env, agents, render=False):
    episode = []
    state = env.get_init_game_state()
    while True:
        #actions = env.get_actions(state) # !! avoids dead agents take action -> gives wrong idea to agent (or) disuades exploration?
        actions = [agent.get_action(state) for agent in agents]
        if render:
            env.render(state)
            print(f"Actions = {[all_actions[a] for a in actions]}")
            time.sleep(1)
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) != 0 else False
        done = done or state.ammo[0] == 0 # !! extra termination rule for 1 v 3 static agents
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
            #running_reward = reward if running_reward is None else ALPHA * running_reward + (1.-ALPHA) * reward
            #running_reward = ALPHA * running_reward + (1.-ALPHA) * reward
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
            print(f"{idx:5d}: win_rate = {total_reward/ params['n_episodes_per_step']:08.7f} - mean length = {len(batch) / params['n_episodes_per_step']:6.2f} - entropy = {stats[0]['entropy']}")


# ------- helper functions -----------------

def process_stats(idx, stats):
    for agent_idx in stats:
        ex.log_scalar(f'loss{agent_idx}', stats[agent_idx]['loss'], step=idx)
        #ex.log_scalar(f'policy_loss{agent_idx}', stats[agent_idx]['policy_loss'], step=idx)
        #ex.log_scalar(f'value_loss{agent_idx}', stats[agent_idx]['value_loss'], step=idx)
        ex.log_scalar(f'grad{agent_idx}', stats[agent_idx]['grads_l2'], step=idx)
        #ex.log_scalar(f'entropy{agent_idx}', stats[agent_idx]['entropy'], step=idx)
        ex.log_scalar(f'grad_var{agent_idx}', stats[agent_idx]['grads_var'], step=idx)

#@ex.automain
def run(params):
    print(params)
    with open(__file__) as f: # print own source code -> easier follow-up in sacred / mongodb
        print(f.read()) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learners = [A2CAgent(idx, device) for idx in [0, 1]]
    others   = [RandomTank(idx) for idx in [2, 3]]
    agents = sorted(learners + others, key=lambda x: x.idx)

    env = Environment(agents, size=params['board_size'],
                        step_penality=params['step_penalty'])
    train(env, learners, others)
    for agent in learners:
        agent.save(f'agent{agent.idx}-temp.pkl')

if __name__ == "__main__" and not STORE:
    run(params)
