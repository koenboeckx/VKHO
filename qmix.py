"""
New implementation of QMIX, with common 
multi-agent controller (MAC)
"""

import random
import copy
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utilities import LinearScheduler, ReplayBuffer, get_args
from new_utilities import generate_episode, Experience, transform_obs, transform_state
from models import QMixModel, QMixForwardModel
from mixers import VDNMixer, QMixer
from env import Agent, Action, Observation
from env import Environment, RestrictedEnvironment

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'QMIX')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))
ex.add_config('default_config.yaml')    # requires PyYAML 
from profilehooks import profile

PRINT = True

class QMIXAgent(Agent):
    def __init__(self, id, team, args):
        super().__init__(id, team)
        self.scheduler = LinearScheduler(start=1.0, stop=0.1,
                                         steps=args.scheduler_steps)
        self.args = args                                        

    def set_model(self, models):
        self.model  = models['model']
    
    def act(self, obs, test_mode=False):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions
                        if action not in unavail_actions]
        
        with torch.no_grad():
            if self.args.model == 'FORWARD':
                qvals = self.model(obs.unsqueeze(0))
            elif self.args.model == 'RNN':
                qvals, self.hidden_state = self.model(obs.unsqueeze(0), self.hidden_state)
            # remove unavailable actions
            for action in unavail_actions:
                qvals[0][action.id] = -np.infty
            action_idx = qvals.max(1)[1].item() # pick position of maximum
        
        if test_mode: # when in test_mode, always return 'best' action
            return self.actions[action_idx]
        
        eps = self.scheduler()
        if random.random() < eps:
            return random.choice(avail_actions)
        else:
            return self.actions[action_idx]
    
    def set_hidden_state(self):
        self.hidden_state = self.model.init_hidden()

class MultiAgentController:
    def __init__(self, env, agents, models, args):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        self.args = args
        if self.args.use_mixer:
            if self.args.mixer == "VDN":
                self.mixer        = VDNMixer()
            elif self.args.mixer == "QMIX":
                self.mixer        = QMixer(args)
            self.target_mixer = copy.deepcopy(self.mixer)
            self.sync_networks()
        self.parameters = list(self.model.parameters())
        if args.use_mixer:
            self.parameters += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr)
        self.normalize_states = args.normalize_states

    def _build_inputs(self, batch):
        agent_idxs = list(range(len(self.agents)))
        states, actions, rewards, next_states, dones, observations,\
            hidden, next_obs, next_hidden, unavailable_actions = zip(*batch)
        
        # transform all into format we require
        states       = torch.stack(states)
        next_states  = torch.stack(next_states)
        if self.normalize_states:
            states = states - states.mean(dim=0)
            next_states = next_states - next_states.mean(dim=0)
        observations = torch.stack(observations)[:, agent_idxs, :]
        next_obs     = torch.stack(next_obs)[:, agent_idxs, :]
        hidden       = torch.stack(hidden).squeeze()[:, agent_idxs, :]
        next_hidden  = torch.stack(next_hidden).squeeze()[:, agent_idxs, :]
        actions      = torch.stack(actions)[:, agent_idxs]
        rewards      = torch.tensor([reward['blue'] for reward in rewards]).unsqueeze(-1)
        dones        = torch.tensor(dones, dtype=torch.float).unsqueeze(-1)
        unavail      = torch.stack(unavailable_actions)[:, agent_idxs, :]
        return states, next_states, observations, next_obs, hidden, next_hidden,\
             actions, rewards, dones, unavail

    def update(self, batch):
        batch_size = len(batch)
        states, next_states, observations, next_obs, hidden, next_hidden, actions,\
            rewards, dones, unavail = self._build_inputs(batch)

        if self.args.model == 'FORWARD':
            current_q_vals   = self.model(observations)
            predicted_q_vals = self.target(next_obs)
        elif self.args.model == 'RNN':
            observations = observations.reshape(batch_size * len(self.agents), -1)
            next_obs = next_obs.reshape(batch_size * len(self.agents), -1)
            hidden = hidden.reshape(batch_size * len(self.agents), -1)
            next_hidden = next_hidden.reshape(batch_size * len(self.agents), -1)
            current_q_vals,   _ = self.model(observations, hidden)
            predicted_q_vals, _ = self.target(next_obs, next_hidden)
            current_q_vals = current_q_vals.reshape(batch_size, len(self.agents), -1)
            predicted_q_vals = predicted_q_vals.reshape(batch_size, len(self.agents), -1)

        # gather q-vals corresponding to the actions taken
        current_q_vals = current_q_vals.reshape(len(self.agents)*batch_size, -1)    # interweave agents: (batch_size x n_agents, n_actions)
        current_q_vals_actions = current_q_vals[range(batch_size * len(self.agents)), actions.reshape(-1)] # select qval for action
        current_q_vals_actions = current_q_vals_actions.reshape(batch_size, -1)     # restore to (batch_size x n_agents)

        predicted_q_vals[unavail==1] = -1e10 # set unavailable actions to low value
        predicted_q_vals_max = predicted_q_vals.max(2)[0]

        if self.args.use_mixer:
            current_q_tot   = self.mixer(current_q_vals_actions, states)
            predicted_q_tot = self.target_mixer(predicted_q_vals_max, next_states)
        else:
            current_q_tot   = current_q_vals_actions
            predicted_q_tot = predicted_q_vals_max

        target = rewards + self.args.gamma * (1. - dones) * predicted_q_tot
        
        # check shapes #TODO: remove when done
        if self.args.use_mixer: # all Q values are squashed into one Qtot
            assert target.shape == torch.Size([self.args.batch_size, 1])
            assert current_q_tot.shape == torch.Size([self.args.batch_size, 1])
        else:
            assert target.shape == torch.Size([self.args.batch_size, len(self.agents)])
            assert current_q_tot.shape == torch.Size([self.args.batch_size, len(self.agents)])
        
        #ex.log_scalar('mean_q', current_q_tot.mean().item())

        td_error = current_q_tot - target.detach()
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.clip)
        self.optimizer.step()

        return loss.item()
    
    def sync_networks(self):
        self.target.load_state_dict(self.model.state_dict())
        if self.args.use_mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        
def generate_models(input_shape, n_actions, args):
    if args.model == 'FORWARD':
        model  = QMixForwardModel(input_shape=input_shape,
                                  n_actions=n_actions,
                                  n_hidden=args.n_hidden)
        target = copy.deepcopy(model)
    elif args.model == 'RNN':
        model  = QMixModel(input_shape=input_shape, n_actions=n_actions, args=args)
        target = QMixModel(input_shape=input_shape, n_actions=n_actions, args=args)
    return {"model": model, "target": target}

PRINT_INTERVAL = 10

def train(args):
    team_blue = [QMIXAgent(idx, "blue", args) for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    if args.env_type == 'normal':
        env = Environment(agents, args)
    if args.env_type == 'restricted':
        env = RestrictedEnvironment(agents, args)

    args.n_actions = 6 + args.n_enemies # 6 fixed actions + 1 aim action per enemy
    args.n_inputs  = 4 + 3*(args.n_friends - 1) + 3*args.n_enemies + args.n_enemies# see process function in models.py
    models = generate_models(args.n_inputs, args.n_actions, args)
    for agent in training_agents:
        agent.set_model(models)

    buffer = ReplayBuffer(size=args.buffer_size)
    mac = MultiAgentController(env, training_agents, models, args)
    for step_idx in range(args.n_steps):
        episode = generate_episode(env, args)
        buffer.insert_list(episode)
        if len(buffer) < args.batch_size:
            continue
        batch = buffer.sample(args.batch_size)
        
        loss = mac.update(batch)

        if step_idx % args.sync_interval == 0:
            mac.sync_networks()
        
        ## logging
        ex.log_scalar('loss', loss)

        if step_idx % args.log_interval == 0:
            episode = generate_episode(env, args, test_mode=True)
            if step_idx == 0:
                episode[-1].rewards["blue"] = 0
                episode[-1].rewards["red"]  = 1
            ex.log_scalar('length', len(episode), step=step_idx)
            ex.log_scalar('reward', episode[-1].rewards["blue"], step=step_idx)
            ex.log_scalar(f'win_blue', int(episode[-1].rewards["blue"] == 1), step=step_idx)
            ex.log_scalar(f'win_red', int(episode[-1].rewards["red"] == 1), step=step_idx)
            ex.log_scalar('epsilon', training_agents[0].scheduler(), step=step_idx)

        if PRINT and step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
            print(f"Step {step_idx}: loss = {loss}, reward = {episode[-1].rewards['blue']}")
            #episode = generate_episode(env, render=True)
        
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
#@profile
def run(_config):
    global args
    args = get_args(_config)
    train(args)
