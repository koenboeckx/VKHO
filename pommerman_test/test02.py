import time, os
import math
from collections import Counter

import numpy as np

import pommerman
from pommerman import agents
import gym

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from pommerman import agents

ROLLOUTS_PER_BATCH = 2

class Agent(agents.BaseAgent):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.states, self.actions, self.hidden, self.values, self.probs = [], [], [], [], []

    def translate_obs(self, o):
        """Transform observation 'o' into something that can be fed to self.model"""
        obs_width = self.model.obs_width
        
        board = o['board'].copy()
        agents = np.column_stack(np.where(board > 10))

        for i, agent in enumerate(agents): 
            agent_id = board[agent[0], agent[1]]
            if agent_id not in o['alive']: # < this fixes a bug >
                board[agent[0], agent[1]] = 0
            else:
                board[agent[0], agent[1]] = 11

        obs_radius = obs_width//2
        pos = np.asarray(o['position'])

        # board
        board_pad = np.pad(board, (obs_radius,obs_radius), 'constant', constant_values=1)
        self.board_cent = board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb blast strength
        bbs = o['bomb_blast_strength']
        bbs_pad = np.pad(bbs, (obs_radius,obs_radius), 'constant', constant_values=0)
        self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb life
        bl = o['bomb_life']
        bl_pad = np.pad(bl, (obs_radius,obs_radius), 'constant', constant_values=0)
        self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        return np.concatenate((
            board_cent, bbs_cent, bl_cent,
            o['blast_strength'], o['can_kick'], o['ammo']), axis=None)
    
    def act(self, obs, action_space):
        """Choose based on obs with SoftMax"""
        obs = self.translate_obs(obs)
        last_hn, last_cn =  (self.hidden[-1][0], # because of use of an LSTM
                            self.hidden[-1][1])
        obs = torch.from_numpy(obs).float().to(self.model.device)
        with torch.no_grad():
            self.model.eval() # disables bathnorm and dropout
            last_hn, last_cn = (torch.tensor(last_hn).unsqueeze(0),
                                torch.tensor(last_cn).unsqueeze(0))
            probs, val, hn, cn = self.model(obs.unsqueeze(0), last_cn, last_hn)
            probs_softmaxed = F.softmax(probs, dim=-1)
            action = Categorical(probs_softmaxed).sample().item()
        self.actions.append(action)
        self.states.append(obs.squeeze(0).numpy())
        self.hidden.append((hn.squeeze(0).clone().detach(),
                            cn.squeeze(0).clone().detach()))
        self.probs.append(probs.detach())                            
        self.values.append(val.detach())

        return action
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.hidden[:]
        del self.values[:]
        del self.probs[:]
        self.hidden.insert(0, self.model.init_rnn())
        
        return self.states, self.actions, self.hidden, self.probs, self.values

class A2CNet(nn.Module):
    def __init__(self, gpu=True):
        super().__init__()
        self.gamma          = .99    # Discount factor for rewards (default 0.99)
        self.entropy_coeff  = 0.01   # Entropy coefficient (0.01)
        self.obs_width = w  = 17     # window height/width (must be odd)
        self.lr             = 0.001  # Learning rate   

        self.inputs_to_conv = ic  = 3*(w**2)    # 3 boards of size w**2
        self.inputs_to_fc   = ifc = 3           # blast strength, can_kick, ammo
        self.conv_channels  = cc  = 45          # number of conv outputs 
        self.flat_after_c   = fac = 13005       # size of flatteded after conv
        
        self.fc1s, self.fc2s, self.fc3s = 1024, 512, 64
        
        self.rnn_input_size     = self.fc2s
        self.rnn_hidden_size    = 64

        self.conv1 = nn.Conv2d(3,  cc, kernel_size=3, stride=1, padding=1, groups=3)
        self.conv2 = nn.Conv2d(cc, cc, kernel_size=3, stride=1, padding=1, groups=3)
        self.conv3 = nn.Conv2d(cc, cc, kernel_size=3, stride=1, padding=1, groups=3)
        self.conv4 = nn.Conv2d(cc, cc, kernel_size=3, stride=1, padding=1, groups=3)

        self.bn1, self.bn2 = nn.BatchNorm2d(cc), nn.BatchNorm2d(cc)
        self.bn3, self.bn4 = nn.BatchNorm2d(cc), nn.BatchNorm2d(cc)

        self.fc_after_conv1 = nn.Linear(fac, self.fc1s)
        self.fc_after_conv2 = nn.Linear(self.fc1s + ifc, self.fc2s)
        self.fc_after_conv3 = nn.Linear(self.fc2s, self.fc2s)
        self.fc_after_conv4 = nn.Linear(self.fc2s, self.fc2s)

        self.rnn = nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size)

        self.fc_after_rnn_1 = nn.Linear(self.rnn_hidden_size, self.fc3s)

        self.action_head = nn.Linear(self.fc3s, 6) # 6 = # of actions
        self.value_head  = nn.Linear(self.fc2s, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.eps       = np.finfo(np.float32).eps.item()

        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda": self.cuda() # !!
        return None# ?why?
    
    def forward(self, x, hn, cn, debug=False):
        batch_size = x.shape[0]
        w, wh = self.obs_width, self.obs_width**2

        boards  = x[:,      0:wh].view(batch_size, 1, w, w)
        bbs     = x[:,   wh:wh*2].view(batch_size, 1, w, w)
        bl      = x[:, wh*2:wh*3].view(batch_size, 1, w, w)

        rest    = x[:, wh*3:]
        to_conv = torch.cat([boards, bbs, bl], 1)

        xc = self.conv1(to_conv)
        xc = self.bn1(xc)
        xc = F.relu(xc)

        xc = self.conv2(xc)
        xc = self.bn2(xc)
        xc = F.relu(xc)

        xc = self.conv3(xc)
        xc = self.bn3(xc)
        xc = F.relu(xc)

        xc = self.conv4(xc)
        xc = self.bn4(xc)
        xc = F.relu(xc) 

        xc = xc.view(batch_size, -1)
        xc = self.fc_after_conv1(xc)
        xc = F.relu(xc)

        xc = torch.cat((xc, rest), 1)
        xc = self.fc_after_conv2(xc)
        xc = F.relu(xc)

        xc = self.fc_after_conv3(xc)
        xc = F.relu(xc)

        xc = self.fc_after_conv4(xc)
        xc = F.relu(xc)

        values = self.value_head(xc)
        hn, cn = self.rnn(xc, (hn, cn)) # RNN is only applied to compute probs
        xc = hn # hidden vector becomes LSTM output

        xc = self.fc_after_rnn_1(xc)
        xc = F.relu(xc)

        probs = self.action_head(xc)

        return probs, values, hn, cn
    
    def init_rnn(self):
        device = self.device
        s = self.rnn_hidden_size
        return (torch.zeros(s).detach().numpy(), torch.zeros(s).detach().numpy())
    
    def discount_rewards(self, _rewards):
        R = 0
        gamma = self.gamma        
        rewards = []
        for r in reversed(_rewards):
            R = r + gamma*R
            rewards.insert(0, R)
        return rewards

class World():
    def __init__(self, init_model=True):
        if init_model:
            self.gmodel = A2CNet(gpu = True) # global model

        self.model = A2CNet(gpu = False)    # Agent (local) model
        self.agent = Agent(self.model)
        self.agent_list = [
            self.agent,
            agents.SimpleAgent()
        ]
        self.env = naked_env(self.agent_list)

def naked_env(agent_list):
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    env._num_items = 0
    env._num_wood  = 0
    env._num_rigid = 0
    env._max_steps = 100

    env.set_init_game_state(None)
    env.set_render_mode('human')
    return env

def do_rollout(env, agent, do_print=False):
    done, state = False, env.reset()
    rewards, dones = [], []
    states, actions, hidden, probs, values = agent.clear()

    while not done:
        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])

        action = env.act(state)
        state, reward, done, info = env.step(action)
        if reward[0] == -1: # stop execution when first agent dies
            done = True
        rewards.append(reward[0])
        dones.append(done)
    
    hidden = hidden[:-1].copy() # copy of agent.hidden -> stores tuples (hn, cn)
    hns, cns = [], []
    #for hns_cns_tuple in hidden:
    #    hns.append(hns_cns_tuple[0])
    #    cns.append(hns_cns_tuple[1])
    for hn, cn in hidden: # by koen
        hns.append(hn)
        cns.append(cn)
    
    return (states.copy(), actions.copy(), # states and actions refer to agent.states and agent.actions
            rewards, dones, (hns, cns),
            probs.copy(), values.copy()) # probs and values refer to agent.probs and agent.values

def gmodel_train(gmodel, states, hns, cns, actions, rewards, gae):
    states, hns, cns = torch.stack(states), torch.stack(hns, dim=0), torch.stack(cns, dim=0)
    gmodel.train() # effect on batchnorm and dropout
    probs, values, _, _ = gmodel(states.to(gmodel.device),
                                hns.to(gmodel.device),
                                cns.to(gmodel.device), debug=False)
    
    prob     = F.softmax(probs, dim=-1)
    log_prob = F.log_softmax(probs, dim=-1)
    entropy  = -(log_prob * prob).sum(1)

    log_probs   = log_prob[range(0, len(actions)), actions]
    advantages  = torch.tensor(rewards).to(gmodel.device) - values.squeeze(1)
    value_loss  = 0.5 * advantages.pow(2)
    policy_loss = -log_probs*torch.tensor(gae).to(gmodel.device) - gmodel.entropy_coeff*entropy

    gmodel.optimizer.zero_grad()
    pl = policy_loss.sum()
    vl = value_loss.sum()
    loss = pl + vl
    loss.backward()
    gmodel.optimizer.step()

    return loss.item(), pl.item(), vl.item()

def unroll_rollouts(gmodel, list_of_full_rollouts):
    gamma = gmodel.gamma
    tau   = 1

    states, actions, rewards, hns, cns, gae = [], [], [], [], [], []
    for (s, a, r, d, h, p, v) in list_of_full_rollouts:
        states.extend(torch.tensor(s))
        actions.extend(a)
        rewards.extend(gmodel.discount_rewards(r))

        hns.extend([torch.tensor(hh) for hh in h[0]])
        cns.extend([torch.tensor(hh) for hh in h[1]])

        # Compute GAE TODO: check this in detail
        last_i, _gae, __gae = len(r)-1, [], 0
        for i in reversed(range(len(r))):
            next_val = v[i+1] if i != last_i else 0
            delta_t = r[i] + gamma * next_val - v[i]
            __gae = __gae * gamma * tau + delta_t
            _gae.insert(0, __gae)
        gae.extend(_gae)
    
    return states, hns, cns, actions, rewards, gae

def train(world: World):
    model, gmodel = world.model, world.gmodel
    agent, env    = world.agent, world.env

    rr = -1
    ii = 0
    for i in range(40000):
        full_rollouts = [do_rollout(env, agent) for _ in range(ROLLOUTS_PER_BATCH)]
        last_rewards = [roll[2][-1] for roll in full_rollouts]
        not_discounted_rewards = [roll[2] for roll in full_rollouts]
        states, hns, cns, actions, rewards, gae = unroll_rollouts(gmodel, full_rollouts)
        gmodel.gamma = .5 + .5 / (1+math.exp(-0.0003*(i-20000))) # adaptive gamma
        l, pl, vl = gmodel_train(gmodel, states, hns, cns, actions, rewards, gae)
        rr = .99 * rr + .01 *  np.mean(last_rewards)/ROLLOUTS_PER_BATCH   
        ii += len(actions)

        print(i, "\t", round(gmodel.gamma, 3), round(rr,3), "\twins:",
              last_rewards.count(1), Counter(actions), round(sum(rewards),3),
              round(l,3), round(pl,3), round(vl,3))
        model.load_state_dict(gmodel.state_dict()) # sync models


    
if __name__ == '__main__':
    world = World(init_model=True)
    train(world)