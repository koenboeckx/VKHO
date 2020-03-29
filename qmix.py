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

PRINT = True

Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations',
    'hidden', 'next_obs', 'next_hidden','unavailable_actions'
])

def transform_obs(observations):
    "transform observation into tensor for storage and use in model"
    result = []
    for agent in observations:
        obs = observations[agent]
        if isinstance(obs, Observation):
            N = 4 + 3 * len(obs.friends) + 3 * len(obs.enemies) + len(obs.enemy_visibility)# own pos (2), friends alive + pos (3*Nf) , friends alive + pos (3*Ne), ammo (1), aim (1), enemy_visibility (Ne)
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
                idx += 3
            # add enemy visibility                
            for visible in obs.enemy_visibility:
                x[idx] = torch.tensor(int(visible))
                idx += 1
            x[idx]   = obs.ammo / args.init_ammo
            x[idx+1] = obs.aim.id if obs.aim is not None else -1
            result.append(x)
        else:
            raise ValueError(f"'obs' should be an Observation, and is a {type(obs)}")
    return torch.stack(result)

def transform_state(state):
    """Transforms State into tensor"""
    # TODO: automate n_enemies calculation -> only valid fot n_enemies = n_friends
    n_agents  = len(state.agents)
    #n_enemies = len(state.visible[state.agents[0]])
    n_enemies = 0
    states_v = torch.zeros(n_agents, 5 + n_enemies) # 5 = x, y, alive, ammo, aim, enemy visible ? (x n_enemies)
    for agent_idx, agent in enumerate(state.agents):
        states_v[agent_idx, 0] = state.position[agent][0] # x
        states_v[agent_idx, 1] = state.position[agent][1] # y
        states_v[agent_idx, 2] = state.alive[agent]
        states_v[agent_idx, 3] = state.ammo[agent] / args.init_ammo
        states_v[agent_idx, 4] = -1 if state.aim[agent] is None else state.aim[agent].id
        #for idx, value in enumerate(state.visible):
        #    states_v[agent_idx, 4+idx] = int(value)
    return states_v

def generate_episode(env, render=False, test_mode=False):
    """Generate episode; store observations and states as tensors
        render: renders every step of episode
        test_mode: picks best action, not based on epsilon-greedy
    """
    episode = []
    state, done = env.reset(), False
    observations = transform_obs(env.get_all_observations())
    n_steps = 0

    for agent in env.agents:        # for agents where it matters,
        agent.set_hidden_state()    # set the init hidden state of the RNN

    while not done:
        unavailable_actions = env.get_unavailable_actions()
        
        # compute action, keep record of hidden state of the agents to store in experience
        actions, hidden, next_hidden = {}, [], []
        for idx, agent in enumerate(env.agents):
            hidden.append(agent.get_hidden_state())
            actions[agent] = agent.act(observations[idx, :], test_mode=test_mode)
            next_hidden.append(agent.get_hidden_state())

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
                                  next_obs, torch.stack(next_hidden),
                                  unavail_actions))
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
    
    def act(self, obs, test_mode=False):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions
                        if action not in unavail_actions]
        
        with torch.no_grad():
            if args.model == 'FORWARD':
                qvals = self.model(obs.unsqueeze(0))
            elif args.model == 'RNN':
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
    def __init__(self, env, agents, models):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        if args.use_mixer:
            if args.mixer == "VDN":
                self.mixer        = VDNMixer()
            elif args.mixer == "QMIX":
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

        if args.model == 'FORWARD':
            current_q_vals   = self.model(observations)
            predicted_q_vals = self.target(next_obs)
        elif args.model == 'RNN':
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

        if args.use_mixer:
            current_q_tot   = self.mixer(current_q_vals_actions, states)
            predicted_q_tot = self.target_mixer(predicted_q_vals_max, next_states)
        else:
            current_q_tot   = current_q_vals_actions
            predicted_q_tot = predicted_q_vals_max

        target = rewards + args.gamma * (1. - dones) * predicted_q_tot
        
        # check shapes #TODO: remove when done
        if args.use_mixer: # all Q values are squashed into one Qtot
            assert target.shape == torch.Size([args.batch_size, 1])
            assert current_q_tot.shape == torch.Size([args.batch_size, 1])
        else:
            assert target.shape == torch.Size([args.batch_size, len(self.agents)])
            assert current_q_tot.shape == torch.Size([args.batch_size, len(self.agents)])
        
        ex.log_scalar('mean_q', current_q_tot.mean().item())

        td_error = current_q_tot - target.detach()
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, args.clip)
        self.optimizer.step()

        return loss.item()
    
    def sync_networks(self):
        self.target.load_state_dict(self.model.state_dict())
        if args.use_mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        
def generate_models(input_shape, n_actions):
    if args.model == 'FORWARD':
        model  = QMixForwardModel(input_shape=input_shape,
                                  n_actions=n_actions,
                                  n_hidden=args.n_hidden)
        target = copy.deepcopy(model)
    elif args.model == 'RNN':
        model  = QMixModel(input_shape=input_shape, n_actions=n_actions)
        target = QMixModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

PRINT_INTERVAL = 10

def train(args):
    team_blue = [QMIXAgent(idx, "blue") for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    if args.env_type == 'normal':
        env = Environment(agents, args)
    if args.env_type == 'restricted':
        env = RestrictedEnvironment(agents, args)

    args.n_actions = 6 + args.n_enemies # 6 fixed actions + 1 aim action per enemy
    args.n_inputs  = 4 + 3*(args.n_friends - 1) + 3*args.n_enemies + args.n_enemies# see process function in models.py
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

        if step_idx % args.sync_interval == 0:
            mac.sync_networks()
        
        ## logging
        ex.log_scalar('loss', loss)

        if step_idx % args.log_interval == 0:
            episode = generate_episode(env, test_mode=True)
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
def run(_config):
    global args
    args = get_args(_config)
    train(args)
