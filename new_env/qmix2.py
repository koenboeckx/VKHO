"""
New implementation of IQL, with one common 
multi-agent controller (MAC)
"""

import random

import numpy as np
import torch
from torch.nn import functional as F

from utilities import LinearScheduler, ReplayBuffer, Experience, generate_episode, get_args
from models import RNNModel
from mixers import QMixer, QMixer_NS
from env import Environment, Agent, Action
from settings import args # to replace with sacred

# TODO: update generate_episode and Experience so that observation is stored as tensor

class QMIXAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        self.scheduler = LinearScheduler(start=1.0, stop=0.1,
                                         steps=args.scheduler_steps)

    def set_model(self, models):
        self.model  = models['model']
    
    def act(self, obs):
        if not obs.alive: # if not alive, do nothing
            return Action(0, 'do_nothing', target=None)

        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in self.actions
                        if action not in unavail_actions]
        
        with torch.no_grad():
            qvals, self.hidden_state = self.model([obs], self.hidden_state)
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


class MultiAgentController:
    def __init__(self, env, agents, models):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        self.mixer  = QMixer_NS()
        parameters = list(self.model.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=args.lr)
        
    def update(self, batch): 
        obs, hidden, next_obs, actions, rewards, dones = self.transform_batch(batch)

        current_q_vals   = torch.zeros((len(obs), args.n_actions))
        predicted_q_vals = torch.zeros((len(obs), args.n_actions))
        for t in range(len(obs)):
            current_q_vals[t, :],   h = self.model([obs[t]], hidden[t])
            predicted_q_vals[t, :], _ = self.model([next_obs[t]], h) # TODO: should be target
        
        current_q_vals_actions = current_q_vals[range(len(obs)), actions]
        current_q_vals_actions = current_q_vals_actions.reshape((-1, len(self.agents)))
        predicted_q_vals_max = predicted_q_vals.max(1)[0].reshape((-1, len(self.agents)))

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
        
    def transform_batch(self, batch):
        """Transforms the inputs of a batch in set of tensors"""
        states, actions, rewards, next_state, dones, observations, hidden, next_obs, unavailable_actions = zip(*batch)
        obs_list = []
        for obs in observations:
            obs_list.extend([obs[agent] for agent in self.agents])
        
        next_obs_list = []
        for obs in next_obs:
            next_obs_list.extend([obs[agent] for agent in self.agents])
        
        hidden_list = []
        for hid in hidden:
            hidden_list.extend([hid[agent] for agent in self.agents])
        
        action_list = []
        for action in actions:
            action_list.extend([action[agent].id for agent in self.agents])

        rewards = torch.tensor([reward["blue"] for reward in rewards])
        dones = torch.tensor(dones, dtype=torch.float)
        
        return obs_list, hidden_list, next_obs_list, action_list, rewards, dones
        

def generate_models(input_shape, n_actions):
    model  = RNNModel(input_shape=input_shape, n_actions=n_actions)
    target = RNNModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

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

        print(f"Step {step_idx}: loss = {loss}, reward = {episode[-1].rewards['blue']}")

if __name__ == '__main__':
    train()
