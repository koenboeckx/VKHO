import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from pommerman import agents

class Agent(agents.BaseAgent):
    def __init__(self, mode):
        super().__init__()
        self.model = model 
        self.states, self.actions, self.hidden, self.values, self.probs = [], [], [], [], []

    def translate_obs(self, o):
        """Transform observation 'o' into something that can be fed to self.model"""
        pass
    
    def act(self, obs, action_space):
        """Choose based on obs with SoftMax"""
        obs = self.translate_obs(obs)
        last_hn, last_cn =  self.hidden[-1][0], # because of use of an LSTM
                            self.hidden[-1][1]
        obs = torch.from_numpy(obs).float().to(self.model.device)
        with torch.no_grad():
            self.model.eval() # disables bathnorm and dropout
            last_hn, last_cn = torch.tensor(last_hn, last_cn).unsqueeze(0)
            probs, val, hn, cn = self.model(obs, last_cn, last_hn)
            probs_softmaxed = F.softmax(probs, dim=-1)
            action = Categorical(probs_softmaxed).sample().item()
        self.actions.append(action)
        self.states.append(obs.squeeze(0).numpy())
        self.hidden.append((hn.squeeze(0).clone().detach(),
                            cn.squeeze(0).clone().detach())
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
        self.value_head  = nn.Linear(self.fc3s, 1)

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
        to_conv = toch.cat([boards, bbs, bl], 1)

        xc = self.conv1(to_conv)
        xc = self.bn1(xc)
        xc = F.relu(xc)

        xc = self.conv2(to_conv)
        xc = self.bn2(xc)
        xc = F.relu(xc)

        xc = self.conv3(to_conv)
        xc = self.bn3(xc)
        xc = F.relu(xc)

        xc = self.conv4(to_conv)
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
        xc = F.relu()

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


