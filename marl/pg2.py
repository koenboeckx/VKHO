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
    ex = Experiment('new_pg')
    ex.observers.append(MongoObserver(url='localhost',
                                    db_name='my_database'))

params = {
    'n_steps':              5000,
    'board_size':           7,
    'gamma':                0.99,
    'learning_rate':        0.001,
    'entropy_beta':         0.01,
    'n_episodes_per_step':  20, # 20
    'init_ammo':            500,
}


class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
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

        self.policy = nn.Linear(128, n_actions)
        self.value  = nn.Linear(128, 1)
    
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
        logits = self.policy(common_out)
        value  = self.value(common_out)
        return value, logits

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

class A2CAgent(Tank):
    def __init__(self, idx, device):
        super().__init__(idx)
        self.device = device
        input_shape = (1, params['board_size'], params['board_size'])
        self.model = A2CModel(input_shape, n_actions=8)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=params['learning_rate'])
    
    def get_action(self, state):
        with torch.no_grad():
            _, logits = self.model(preprocess([state]))
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

    def update_a2c(self, batch):
        """Perform true A2C update."""
 
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        own_rewards = [reward[self.idx] for reward in rewards]
        actions_v = torch.LongTensor(own_actions)
        rewards_v = torch.Tensor(own_rewards)
        dones_v   = torch.FloatTensor(dones)

        values_v, logits_v = self.model(preprocess(states))
        values_v = values_v.squeeze()
        next_values_v, _ = self.model(preprocess(next_states))
        next_values_v = next_values_v.squeeze().detach() # no learning for these states
        returns_v = (rewards_v + params['gamma'] * (1. - dones_v) * next_values_v).detach()
        advantages_v = returns_v - values_v
        
        ## discounting ...
        #for ii in range(len(values_v)-1, -1, -1):
        #    values_v[ii] *= params['gamma']**(len(values_v)-ii+1)
        #    predicted_v[ii] *= params['gamma']**(len(predicted_v)-ii+1)
        
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = advantages_v.detach() * logprob_actions_v
        policy_loss = -logprob_act_vals_v.mean()

        value_loss = advantages_v.pow(2).mean()
        
        probs_v = F.softmax(logits_v, dim=1)
        entropy = -(probs_v * logprobs_v).sum(dim=1)
        
        loss = policy_loss + value_loss - params['entropy_beta'] * entropy.mean()
        loss.backward()

        stats = self._create_stats(loss, policy_loss, value_loss, entropy)

        self.optimizer.step()
        return stats

    def update_standard(self, batch):
        self.optimizer.zero_grad()

        states, actions, _, _, _ = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.LongTensor(own_actions)

        discounted = self.discount_rewards(batch)
        returns_v = torch.tensor(discounted)
        #returns_v = torch.tensor(self.discount_rewards(batch))

        values_v, logits_v = self.model(preprocess(states))
        values_v = values_v.squeeze()
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = (returns_v - values_v.detach()) * logprob_actions_v
        policy_loss = -logprob_act_vals_v.mean()

        value_loss = F.mse_loss(returns_v, values_v)
        
        probs_v = F.softmax(logits_v, dim=1)
        entropy = -(probs_v * logprobs_v).sum(dim=1)
        
        loss = policy_loss + value_loss - params['entropy_beta'] * entropy.mean()
        loss.backward()

        stats = self._create_stats(loss, policy_loss, value_loss, entropy)

        self.optimizer.step()
        return stats

    def update_pg(self, batch):
        self.optimizer.zero_grad()

        states, actions, _, _, _ = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.LongTensor(own_actions)

        returns = self.discount_rewards(batch)
        returns_v = torch.tensor(returns)

        _, logits_v = self.model(preprocess(states))
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = returns_v * logprob_actions_v
        loss = -logprob_act_vals_v.mean()

        loss.backward()
        
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
        }
        self.optimizer.step()
        return stats

    def update(self, batch): # TODO: don't forget
        return self.update_pg(batch)

    def _create_stats(self, loss, policy_loss, value_loss, entropy):
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])
        stats = {
            'loss':         loss.item(),
            'policy_loss':  policy_loss.item(),
            'value_loss':   value_loss.item(),
            'grads_l2':     np.sqrt(np.mean(np.square(grads))),
            'entropy':      entropy.mean().item(),
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
        actions = [agent.get_action(state) for agent in agents]
        if render:
            env.render(state)
            print(f"Actions = {[all_actions[a] for a in actions]}")
            time.sleep(1)
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) != 0 else False
        episode.append(Experience(state, actions, reward, next_state, done))
        if done:
            return episode
        state = next_state

def train(env, learners, others):
    stats, grads = {}, {}
    for agent in learners:
        grads[agent.idx] = []
    running_reward = 0

    agents = sorted(learners + others, key=lambda x: x.idx)
    for idx in range(params['n_steps']):
        batch = []
        for epi_idx in range(params['n_episodes_per_step']):
            render = True if DEBUG and epi_idx % 10 == 0 else False
            episode = play_episode(env, agents, render=render)
            reward  = episode[-1].reward[0] 
            #running_reward = reward if running_reward is None else ALPHA * running_reward + (1.-ALPHA) * reward
            running_reward = ALPHA * running_reward + (1.-ALPHA) * reward

            batch.extend(episode)
        if STORE:
            ex.log_scalar('mean_length', len(batch) / params['n_episodes_per_step'])
        
        for agent in learners:
            stats[agent.idx] = agent.update(batch)
            grads[agent.idx].append(stats[agent.idx]['grads_l2'])
        process_stats(idx, running_reward, stats)
    if STORE:
        log_grad_variance(learners, grads)

def log_grad_variance(learners, grads):    
    """Compute and log variance of gradients of all learning agents"""
    for agent in learners:
        variance = compute_grad_variance(grads[agent.idx])
        for idx, var in enumerate(variance):
            ex.log_scalar(f'var_grad{agent.idx}', var, step=idx)


# ------- helper functions -----------------

def process_stats(idx, reward, stats):
    if STORE:
        ex.log_scalar('reward', reward, step=idx)
        for agent_idx in stats:
            ex.log_scalar(f'loss{agent_idx}', stats[agent_idx]['loss'], step=idx)
            #ex.log_scalar(f'policy_loss{agent_idx}', stats[agent_idx]['policy_loss'], step=idx)
            #ex.log_scalar(f'value_loss{agent_idx}', stats[agent_idx]['value_loss'], step=idx)
            ex.log_scalar(f'grad{agent_idx}', stats[agent_idx]['grads_l2'], step=idx)
            #ex.log_scalar(f'entropy{agent_idx}', stats[agent_idx]['entropy'], step=idx)
    print(f"{idx:5d}: running reward = {reward:08.7f}")

def compute_grad_variance(grads):
    variance = []
    for idx in range(5, len(grads)-5):
        grad_var = np.var(grads[idx-5:idx+5])
        variance.append(grad_var)
    return variance

@ex.automain
def run():
    print(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learners = [A2CAgent(idx, device) for idx in [0, 1]]
    others   = [RandomTank(idx) for idx in [2, 3]]
    agents = sorted(learners + others, key=lambda x: x.idx)

    env = Environment(agents, size=params['board_size'])
    train(env, learners, others)
    for agent in learners:
        agent.save(f'agent{agent.idx}-temp.pkl')


if __name__ == "__main__" and not STORE:
    run()
