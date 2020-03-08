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
from models import QMixModel
from mixers import VDNMixer, QMixer
from env import Environment, Agent, Action, Observation

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'QMIX')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))
ex.add_config('new_env/default_config.yaml')    # requires PyYAML 

PRINT = True

def transform_obs(observations):
    "transform observation into tensor for storage and use in model"
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
    "Generate episode; store observations and states as tensors"
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
        unavail_actions = torch.zeros((args.n_agents, args.n_actions), dtype=torch.long)
        for idx, agent in enumerate(env.agents):
            act_ids = [act.id for act in unavailable_actions[agent]]
            unavail_actions[idx, act_ids] = 1.

        episode.append(Experience(transform_state(state), actions, rewards, 
                                  transform_state(next_state), done, 
                                  observations, torch.stack(hidden), 
                                  next_obs, unavail_actions))
        state = next_state
        observations = next_obs
    
    if render:
        print(f"Game won by team {env.terminal(next_state)}")
    return episode

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
            action_idx = qvals.max(1)[1].item() # pick position of maximum

        eps = self.scheduler()
        if random.random() < eps:
            return random.choice(avail_actions)
        else:
            return self.actions[action_idx]
    
    def set_hidden_state(self):
        self.hidden_state = self.model.init_hidden()

class MultiAgentController:
    def __init__(self, env, agents, models):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        if args.use_mixer:
            if args.mixer == "VDN":
                self.mixer        = VDNMixer()
                self.target_mixer = VDNMixer()
            elif args.mixer == "QMIX":
                self.mixer        = QMixer(args)
                self.target_mixer = QMixer(args)
            self.sync_networks()
        parameters = list(self.model.parameters())
        if args.use_mixer:
            parameters += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=args.lr)
        
    def update(self, batch): 
        #obs, hidden, next_obs, actions, rewards, dones = self.transform_batch(batch)
        agent_idxs = list(range(len(self.agents)))
        batch_size = len(batch)
        states, actions, rewards, next_states, dones, observations, hidden, next_obs, unavailable_actions = zip(*batch)
        
        states       = torch.stack(states)
        next_states  = torch.stack(next_states)
        observations = torch.stack(observations)[:, agent_idxs, :]
        next_obs     = torch.stack(next_obs)[:, agent_idxs, :]
        hidden       = torch.stack(hidden)[:, agent_idxs, :]
        actions      = torch.stack(actions)[:, agent_idxs]
        rewards      = torch.tensor([reward['blue'] for reward in rewards]).unsqueeze(-1)
        dones        = torch.tensor(dones, dtype=torch.float).unsqueeze(-1)
        unavail      = torch.stack(unavailable_actions)[:, agent_idxs, :]

        current_q_vals   = torch.zeros((batch_size, len(self.agents), args.n_actions))
        predicted_q_vals = torch.zeros((batch_size, len(self.agents), args.n_actions))
        for t in range(batch_size):
            current_q_vals[t, :, :],    h = self.model(observations[t], hidden[t])
            predicted_q_vals[t, :,  :], _ = self.target(next_obs[t], h) 

        # gather q-vals corresponding to the actions taken
        current_q_vals = current_q_vals.reshape(len(self.agents)*batch_size, -1)    # interweave agents: (batch_size x n_agents, n_actions)
        current_q_vals_actions = current_q_vals[range(batch_size * len(self.agents)), actions.reshape(-1)] # select qval for action
        current_q_vals_actions = current_q_vals_actions.reshape(batch_size, -1)     # restore to (batch_size x n_agents)

        predicted_q_vals[unavail==1] = -1e10 # set unavailable actions to low value
        predicted_q_vals_max = predicted_q_vals.max(2)[0]

        if args.use_mixer:
            current_q_tot   = self.mixer(current_q_vals_actions, states)
            predicted_q_tot = self.mixer(predicted_q_vals_max, next_states)
        else:
            current_q_tot   = current_q_vals_actions
            predicted_q_tot = predicted_q_vals_max

        target = rewards + args.gamma * (1. - dones) * predicted_q_tot
        
        # check shapes #TODO: remove when done
        if args.use_mixer:
            assert target.shape == torch.Size([args.batch_size, 1])
            assert current_q_tot.shape == torch.Size([args.batch_size, 1])
        else:
            assert target.shape == torch.Size([args.batch_size, len(self.agents)])
            assert current_q_tot.shape == torch.Size([args.batch_size, len(self.agents)])

        loss = F.mse_loss(current_q_tot, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def sync_networks(self):
        self.target.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
def generate_models(input_shape, n_actions):
    model  = QMixModel(input_shape=input_shape, n_actions=n_actions)
    target = QMixModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

PRINT_INTERVAL = 10

def train(args):
    team_blue = [QMIXAgent(idx, "blue") for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents, args)

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

        ex.log_scalar('length', len(episode))
        ex.log_scalar('reward', episode[-1].rewards["blue"])
        ex.log_scalar(f'win_blue', int(episode[-1].rewards["blue"] == 1))
        ex.log_scalar(f'win_red', int(episode[-1].rewards["red"] == 1))
        ex.log_scalar('loss', loss)
        ex.log_scalar('epsilon', training_agents[0].scheduler())

        if PRINT and step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
            print(f"Step {step_idx}: loss = {loss}, reward = {episode[-1].rewards['blue']}")
            #episode = generate_episode(env, render=True)
        
        if args.use_mixer and step_idx % args.sync_interval == 0:
            if PRINT: print('Syncing networks ...')
            mac.sync_networks()
        
        if args.save_model and step_idx > 0 and step_idx % args.save_model_interval == 0:
            from os.path import expanduser
            home = expanduser("~")
            torch.save(models["model"].state_dict(), home+args.path+f'RUN_{get_run_id()}_MODEL.torch')
            if args.use_mixer:
                torch.save(mac.mixer.state_dict(), home+args.path+f'RUN_{get_run_id()}_MIXER.torch')

#--------------------------------------------------------------------        
@ex.capture
def get_run_id(_run): # enables saving model with run id
    return _run._id

@ex.automain
def run(_config):
    global args
    args = get_args(_config)
    train(args)
