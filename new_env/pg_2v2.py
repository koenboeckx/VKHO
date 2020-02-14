from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import Experience, generate_episode

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('PG-2v2')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

class PGModel(nn.Module):   # Q-Learning Model
    def __init__(self, input_shape, n_hidden, n_actions, lr):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
    
    def forward(self, x):
        x = self.process(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def process(self, obs_list):
        "transform list of observations into tensor"
        x = torch.zeros((len(obs_list), 13)) # own pos (2), friends alive + pos (3) , friends alive + pos (2x3), ammo (1), aim (1)
        if isinstance(obs_list[0], Observation):
            for obs_idx, obs in enumerate(obs_list):
                x[obs_idx, 0:2] = torch.tensor(obs.own_position)
                friend_pos = obs.friends[0] # TODO: improve below to allow more friends/enemies
                if friend_pos:     
                    x[obs_idx, 2:5] = torch.tensor([1.,] + list(obs.friends[0]))
                else:
                    x[obs_idx, 2:5] = torch.tensor([0., 0., 0.])
                enemy1_pos = obs.enemies[0]
                if enemy1_pos:
                    x[obs_idx, 5:8] = torch.tensor([1.,] + list(obs.enemies[0]))
                else:
                    x[obs_idx, 5:8] = torch.tensor([0., 0., 0.])
                enemy2_pos = obs.enemies[1]
                if enemy2_pos:
                    x[obs_idx, 8:11] = torch.tensor([1.,] + list(obs.enemies[1]))
                else:
                    x[obs_idx, 8:11] = torch.tensor([0., 0., 0.])

                x[obs_idx, 11]   = obs.ammo / params['init_ammo']
                x[obs_idx, 12]   = int(obs.aim is not None)
        else:
            raise ValueError((f"x should be (list of) Observation(s)"))
        return x

class PGAgent(Agent):
    def __init__(self, id, team, model, params):
        super().__init__(id, team, params)
        self.model = model
    
    def act(self, obs):
        if not obs.alive: # if not alive, do nothing
            return 0
        unavail_actions = self.env.get_unavailable_actions()[self]
        with torch.no_grad():
            logits = self.model([obs])[0]
            for action in unavail_actions:
                logits[action] = -np.infty
            action = Categorical(logits=logits).sample().item()
        return action
    
    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + params['gamma'] * R
            returns.insert(0, R)
        return returns

    def update(self, batch):
        states, actions, rewards, _, _, observations, _, unavail = zip(*batch) # TODO: use unavail? -> actions not allowed aren't used
        
        # only perform updates on actions performed while alive
        self_alive_idx = [idx for idx, obs in enumerate(observations) if obs[self].alive]
        observations = [observations[idx][self] for idx in self_alive_idx]
        actions = torch.tensor([action[self] for action in actions])[self_alive_idx]

        rewards = [reward[self] for reward in rewards]
        expected_returns = torch.tensor(self.compute_returns(rewards))[self_alive_idx]

        logits = self.model(observations)
        for idx in self_alive_idx:
            unavail_actions = unavail[idx][self]
            logits[idx][unavail_actions] = -999

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob_act = log_prob[self_alive_idx, actions]
        #log_prob_act_val = expected_returns * log_prob_act
        log_prob_act_val = (expected_returns - expected_returns.mean()) * log_prob_act
        loss = -log_prob_act_val.mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        return loss.item()

params = {
    'board_size':           5,
    'init_ammo':            5,
    'max_range':            3,
    'step_penalty':         0.01,
    'max_episode_length':   100,
    'gamma':                0.99,
    'n_hidden':             128,

    'n_steps':              10000,
    'lr':                   0.0001,
}

@ex.config
def cfg():
    params = params

PRINT_INTERVAL = 100

@ex.automain
def run(params):
    model = PGModel(input_shape=13, n_hidden=params["n_hidden"],
                    n_actions=len(all_actions), lr=params["lr"])

    team_blue = [PGAgent(0, "blue", model, params), PGAgent(1, "blue", model, params)]
    team_red  = [Agent(2, "red", params),  Agent(3, "red", params)]

    training_agents = team_blue
    agents = team_blue + team_red

    env = Environment(agents, params)
    epi_len, nwins = 0, 0
    for step_idx in range(params["n_steps"]):
        episode = generate_episode(env)

        epi_len += len(episode)
        reward = episode[-1].rewards[env.agents[0]]

        ex.log_scalar('length', len(episode), step=step_idx)
        ex.log_scalar('reward', reward, step=step_idx)
        ex.log_scalar(f'win', int(episode[-1].rewards[agents[0]] == 1), step=step_idx)

        if episode[-1].rewards[agents[0]] == 1:
            nwins += 1

        for agent in training_agents:
            loss = agent.update(episode)
        ex.log_scalar('loss', loss, step=step_idx)
        
        if step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
            s  = f"Step {step_idx}: loss: {loss:8.4f} - "
            s += f"Average length: {epi_len/PRINT_INTERVAL:5.2f} - "
            s += f"win ratio: {nwins/PRINT_INTERVAL:4.3f} - "
            print(s)
            epi_len, nwins = 0, 0