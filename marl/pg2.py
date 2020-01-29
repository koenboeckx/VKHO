

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
    'n_steps':              200,
    'board_size':           7,
    'gamma':                0.99,
    'learning_rate':        0.001,
    'entropy_beta':         0.01,
    'n_episodes_per_step':  40, # 20
    'init_ammo':            5,
    'step_penalty':         0.05, # to induce shorter episodes
}

@ex.config
def cfg():
    params = params

# ----------------------- Models -----------------------------------------

class A2CCommonModel(nn.Module):
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

class A2CModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        self.common = A2CCommonModel(input_shape)

        self.policy = nn.Linear(128, n_actions)
        self.value  = nn.Linear(128, 1)

    def forward(self, x):
        """Input x has two parts: a 'board' for the conv layer,
        and an 'other', containing ammo and alive flags for
        the fully connecte layer."""
        common_out = self.common(x)
        logits = self.policy(common_out)
        value  = self.value(common_out)
        return value, logits

class A2CGRUModel(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=512, num_layers=1):
        super().__init__()
        self.common = A2CCommonModel(input_shape)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(128, hidden_size, num_layers=num_layers)

        self.policy = nn.Linear(hidden_size, n_actions)
        self.value  = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        common_out = self.common(x)
        hidden_state = self.init_hidden(batch_size)
        rnn_out, hidden_state = self.gru(common_out.unsqueeze(0), hidden_state)
        logits = self.policy(hidden_state).squeeze()
        value  = self.value(hidden_state).squeeze() # TODO: check if this 'squeeze' is necessary !!
        return value, logits

# ----------------------- Agents -----------------------------------------
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
        #
        #self.model = A2CModel(input_shape, n_actions=8)
        self.model = A2CGRUModel(input_shape, n_actions=8)
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
        next_values_v = next_values_v.squeeze() # no learning for these states
        td_target_v = rewards_v + params['gamma'] * (1. - dones_v) * next_values_v.detach()
        delta_v = td_target_v - values_v
        
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = delta_v.detach() * logprob_actions_v
        policy_loss = -logprob_act_vals_v.mean()

        value_loss = F.smooth_l1_loss(values_v, td_target_v.detach())
        
        probs_v = F.softmax(logits_v, dim=1)
        entropy_loss = (probs_v * logprobs_v).sum(dim=1).mean()
        
        loss = policy_loss + value_loss + params['entropy_beta'] * entropy_loss
        loss.backward()

        stats = self._create_stats(loss, policy_loss, value_loss, entropy_loss)

        self.optimizer.step()
        return stats

    def update_pg_baseline(self, batch):
        self.optimizer.zero_grad()

        states, actions, _, _, _ = zip(*batch)
        
        own_actions = [action[self.idx] for action in actions]
        actions_v = torch.LongTensor(own_actions)

        discounted = self.discount_rewards(batch)
        returns_v = torch.tensor(discounted)

        values_v, logits_v = self.model(preprocess(states))
        values_v = values_v.squeeze()
        logprobs_v = F.log_softmax(logits_v, dim=-1)
        logprob_actions_v  = logprobs_v[range(len(batch)), actions_v]
        logprob_act_vals_v = (returns_v - values_v.detach()) * logprob_actions_v
        policy_loss = -logprob_act_vals_v.mean()

        value_loss = F.mse_loss(returns_v.detach(), values_v)
        
        probs_v = F.softmax(logits_v, dim=1)
        entropy = -(probs_v * logprobs_v).sum(dim=1).mean()
        
        loss = policy_loss + value_loss # - params['entropy_beta'] * entropy
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
        self.optimizer.step()
        return stats

    def update(self, batch):
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
            print(f"{idx:5d}: win_rate = {total_reward/ params['n_episodes_per_step']:08.7f} - mean length = {len(batch) / params['n_episodes_per_step']:6.2f}")

# ------- helper functions -----------------

def process_stats(idx, stats):
    for agent_idx in stats:
        ex.log_scalar(f'loss{agent_idx}', stats[agent_idx]['loss'], step=idx)
        ex.log_scalar(f'policy_loss{agent_idx}', stats[agent_idx]['policy_loss'], step=idx)
        ex.log_scalar(f'value_loss{agent_idx}', stats[agent_idx]['value_loss'], step=idx)
        ex.log_scalar(f'grad{agent_idx}', stats[agent_idx]['grads_l2'], step=idx)
        ex.log_scalar(f'entropy{agent_idx}', stats[agent_idx]['entropy'], step=idx)
        ex.log_scalar(f'grad_var{agent_idx}', stats[agent_idx]['grads_var'], step=idx)

@ex.automain
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
