"""
New implementation of IQL, with one common 
multi-agent controller (MAC)
"""

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utilities import LinearScheduler, ReplayBuffer, Experience, get_args
from models import RNNModel
from mixers import QMixer, QMixer_NS
from env import Environment, Agent, Action, Observation
from settings import args # to replace with sacred

# TODO: update generate_episode and Experience so that observation is stored as tensor

class QMixModel(nn.Module): # TODO: add last action as input
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.rnn_hidden_dim = args.n_hidden
        self.fc1 = nn.Linear(input_shape, args.n_hidden) 
        self.rnn = nn.GRUCell(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = inputs
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QMixer(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Hypernetwork
        self.n_trainers = args.n_friends # assumes all friends are learning
        self.embed_dim = embed_dim
        self.state_dim = 5 * args.n_agents # every agent is represented by 5 values: x, y, alive, ammo, aim
        self.HW1 = nn.Linear(self.state_dim, embed_dim * self.n_trainers)
        self.Hb1 = nn.Linear(self.state_dim, embed_dim)
        self.HW2 = nn.Linear(self.state_dim, embed_dim)
        self.Hb2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        agent_qs = process_qs(agent_qs).unsqueeze(2) # add 3rd dimension: (bs x n_trainers x 1)
        states = process_states(states)

        # computes matrices via hypernetwork
        states = states.reshape(-1, self.state_dim)
        W1 = torch.abs(self.HW1(states))
        W1 = W1.reshape(-1, self.embed_dim, self.n_trainers)
        
        b1 = self.Hb1(states)
        b1 = b1.reshape(-1, self.embed_dim, 1)
        
        W2 = torch.abs(self.HW2(states))
        W2 = W2.reshape(-1, 1, self.embed_dim)
        
        b2 = F.relu(self.Hb2(states))
        b2 = b2.reshape(-1, 1, 1)

        # real network updates
        QW1 = torch.bmm(W1, agent_qs)   # (bs x embed_dim x 1)
        Qb1 = F.elu(QW1 + b1)           # (bs x embed_dim x 1)
        QW2 = torch.bmm(W2, Qb1)        # (bs x 1 x 1)
        Qtot = QW2 + b2                 # (bs x 1 x 1)
        return Qtot.squeeze()           # (bs)

class QMIXAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        self.scheduler = LinearScheduler(start=1.0, stop=0.1,
                                         steps=args.scheduler_steps)

    def set_model(self, models):
        self.model  = models['model']
    
    def act(self, obs):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions
                        if action not in unavail_actions]
        
        with torch.no_grad():
            qvals, self.hidden_state = self.model(obs.unsqueeze(0), self.hidden_state)
            # remove unavailable actions
            for action in unavail_actions:
                qvals[0][action.id] = -np.infty
            action_idx = qvals.max(1)[1].item()

        eps = self.scheduler()
        if random.random() < eps:
            return random.choice(avail_actions)
        else:
            return self.actions[action_idx]
    
    def set_hidden_state(self):
        self.hidden_state = self.model.init_hidden()

def transform_obs(observations):
    "transform lobservation into tensor for storage and use in model"
    result = []
    for agent in observations:
        obs = observations[agent]
        if isinstance(obs, Observation):
            N = 4 + 3 * len(obs.friends) + 3 * len(obs.enemies) # own pos (2), friends alive + pos (3*Nf) , friends alive + pos (3*Ne), ammo (1), aim (1)
            x = torch.zeros((N)) 
            x[0:2] = torch.tensor(obs.own_position)
            idx = 2
            for friend in obs.friends:
                if friend:  # friend is alive
                    x[idx:idx+3] = torch.tensor([1.,] + list(friend))
                else:       # friend is dead
                    x[idx:idx+3] = torch.tensor([0., 0., 0.])
                idx += 3
            for enemy in obs.enemies:
                if enemy:   # enemy is alive
                    x[idx:idx+3] = torch.tensor([1.,] + list(enemy))
                else:       # enemy is dead
                    x[idx:idx+3] = torch.tensor([0., 0., 0.])
                idx += 1
            x[idx]   = obs.ammo / args.init_ammo
            x[idx+1] = obs.aim.id if obs.aim is not None else -1
            result.append(x)
        else:
            raise ValueError(f"'obs' should be an Observation, and is a {type(obs)}")
    return torch.stack(result)

def transform_state(state):
    """Transforms State into tensor"""
    n_agents = len(state.agents)
    states_v = torch.zeros(n_agents, 5) # 5 = x, y, alive, ammo, aim
    for agent_idx, agent in enumerate(state.agents):
        states_v[agent_idx, 0] = state.position[agent][0] # x
        states_v[agent_idx, 1] = state.position[agent][1] # y
        states_v[agent_idx, 2] = state.alive[agent]
        states_v[agent_idx, 3] = state.ammo[agent] / args.init_ammo
        states_v[agent_idx, 4] = -1 if state.aim[agent] is None else state.aim[agent].id
    return states_v

def generate_episode(env, render=False):
    episode = []
    state, done = env.reset(), False
    observations = transform_obs(env.get_all_observations())
    n_steps = 0

    for agent in env.agents:        # for agents where it matters,
        agent.set_hidden_state()    # set the init hidden state of the RNN

    while not done:
        unavailable_actions = env.get_unavailable_actions()
        
        # keep record of hidden state of the agents to store in experience
        hidden = []
        for agent in env.agents:
            hidden.append(agent.get_hidden_state())
        
        #actions = env.act(observations)
        actions = {}
        for idx, agent in enumerate(env.agents):
            actions[agent] = agent.act(observations[idx, :])

        if render:
            print(f"Step {n_steps}")
            env.render()
            print([action.name for action in actions.values()])

        next_state, rewards, done, _ = env.step(actions)
        next_obs = transform_obs(env.get_all_observations())
        
        # episodes that take long are not allowed and penalized for both agents
        n_steps += 1
        if n_steps > args.max_episode_length:
            done = True
            rewards = {'blue': -1, 'red': -1}

        actions = torch.tensor([action.id for action in actions.values()])
        episode.append(Experience(transform_state(state), actions, rewards, 
                                  transform_state(next_state), done, 
                                  observations, torch.stack(hidden), 
                                  next_obs, unavailable_actions))
        state = next_state
        observations = next_obs
    
    if render:
        print(f"Game won by team {env.terminal(next_state)}")
    return episode

class MultiAgentController:
    def __init__(self, env, agents, models):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        self.mixer  = QMixer()
        parameters = list(self.model.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=args.lr)
        
    def update(self, batch): 
        #obs, hidden, next_obs, actions, rewards, dones = self.transform_batch(batch)
        agent_idxs = list(range(len(self.agents)))
        batch_size = len(batch)
        states, actions, rewards, next_state, dones, observations, hidden, next_obs, unavailable_actions = zip(*batch)
        states = torch.stack(states)

        observations = torch.stack(observations)[:, agent_idxs, :]
        next_obs     = torch.stack(next_obs)[:, agent_idxs, :]
        hidden       = torch.stack(hidden)[:, agent_idxs, :]
        actions      = torch.stack(actions)[:, agent_idxs]
        rewards      = torch.tensor([reward['blue'] for reward in rewards])
        dones        = torch.tensor(dones, dtype=torch.float)

        current_q_vals   = torch.zeros((batch_size, len(self.agents), args.n_actions))
        predicted_q_vals = torch.zeros((batch_size, len(self.agents), args.n_actions))
        for t in range(batch_size):
            current_q_vals[t, :, :],    h = self.model(observations[t], hidden[t])
            predicted_q_vals[t, :,  :], _ = self.model(next_obs[t], h) # TODO: should be target
        
        # gather q-vals corresponding to the actions taken
        current_q_vals_actions = current_q_vals.reshape(2*batch_size, -1)[range(batch_size * len(self.agents)), actions.reshape(-1)]
        current_q_vals_actions = current_q_vals_actions.reshape(batch_size, -1)
        predicted_q_vals_max = predicted_q_vals.max(2)[0]

        # TODO: do states
        states, next_states = None, None

        current_q_tot   = self.mixer(current_q_vals_actions, states)
        predicted_q_tot = self.mixer(predicted_q_vals_max, next_states)
        target = rewards + args.gamma * (1. - dones) * predicted_q_tot
        
        loss = F.mse_loss(current_q_tot, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        

def generate_models(input_shape, n_actions):
    model  = QMixModel(input_shape=input_shape, n_actions=n_actions)
    target = QMixModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

PRINT_INTERVAL = 10

def train():
    team_blue = [QMIXAgent(idx, "blue") for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)

    args.n_actions = 6 + args.n_enemies # 6 fixed actions + 1 aim action per enemy
    args.n_inputs  = 4 + 3*(args.n_friends - 1) + 3*args.n_enemies # see process function in models.py
    models = generate_models(args.n_inputs, args.n_actions)
    for agent in training_agents:
        agent.set_model(models)

    buffer = ReplayBuffer(size=args.buffer_size)
    mac = MultiAgentController(env, training_agents, models)
    for step_idx in range(args.n_steps):
        episode = generate_episode(env)
        buffer.insert_list(episode)
        if len(buffer) < args.batch_size:
            continue
        batch = buffer.sample(args.batch_size)
        
        loss = mac.update(batch)

        if step_idx > 0 and step_idx%PRINT_INTERVAL == 0:
            print(f"Step {step_idx}: loss = {loss}, reward = {episode[-1].rewards['blue']}")
        
if __name__ == '__main__':
    train()
