# TODO: ablations
#           * QMIX-NS : QMIX without hypernetworks 


import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env import Action, Agent, Environment
from utilities import LinearScheduler, ReplayBuffer, Experience, generate_episode
from models import ForwardModel, RNNModel
from settings import args, Args
from mixers import VDNMixer, QMixer, QMixer_NS

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'QMIX-{args.n_friends}v{args.n_enemies}')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

class QMixAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        self.scheduler = args.scheduler(start=1.0, stop=0.1,
                                        steps=args.scheduler_steps)
        
    def set_model(self, models):
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()

    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
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

    def compute_q(self, batch):
        # TODO: exclude actions when not alive ??
        _, actions, rewards, _, dones, observations, hidden, next_obs, unavailable_actions = zip(*batch)
        actions = [action[self].id for action in actions]
        rewards = torch.tensor([reward[self.team] for reward in rewards])
        observations = [obs[self] for obs in observations]
        next_obs = [obs[self] for obs in next_obs]
        hidden = [hidden_state[self] for hidden_state in hidden]
        dones = torch.tensor(dones, dtype=torch.float)
        
        current_qvals   = torch.zeros(len(batch), args.n_actions)
        predicted_qvals = torch.zeros(len(batch), args.n_actions)
        for t in range(len(batch)):
            current_qvals[t, :],   h = self.model([observations[t]], hidden[t])
            predicted_qvals[t, :], _ = self.model([next_obs[t]], h)

        current_qvals_actions = current_qvals[range(len(batch)), actions]
        
        # Set unavailable action to very low Q-value !!
        for idx in range(len(unavailable_actions)):
            unavail_actions = unavailable_actions[idx][self]
            unavail_ids = [action.id for action in unavail_actions]
            predicted_qvals[idx][unavail_ids] = -np.infty

        predicted_qvals_max = predicted_qvals.max(1)[0]

        return current_qvals_actions, predicted_qvals_max
    

def generate_models(input_shape, n_actions):
    model  = RNNModel(input_shape=input_shape, n_actions=n_actions)
    target = RNNModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

def train(args):
    team_blue = [QMixAgent(idx, "blue") for idx in range(args.n_friends)] # TODO: args.n_friends should be 2 when 2 agents in team "blue"
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)

    args.n_actions = 6 + args.n_enemies # 6 fixed actions + 1 aim action per enemy
    args.n_inputs  = 4 + 3*(args.n_friends-1) + 3*args.n_enemies # see process function in models.py
    models = generate_models(args.n_inputs, args.n_actions)
    for agent in training_agents:
        agent.set_model(models)
    
    # configure mixers
    if args.mixer == 'VDN':
        mixer = VDNMixer()
        target_mixer = VDNMixer()
    elif args.mixer == 'QMIX':
        mixer = QMixer()
        target_mixer = QMixer()
    elif args.mixer == 'QMIX_NS':
        mixer = QMixer_NS()
        target_mixer = QMixer_NS()

    buffer = ReplayBuffer(size=args.buffer_size)
    epi_len, nwins = 0, 0

    ex.log_scalar(f'win', 0.0, step=1) # forces start of run at 0 wins ()
    for step_idx in range(args.n_steps):
        episode = generate_episode(env)
        buffer.insert_list(episode)
        if not buffer.can_sample(args.batch_size):
            continue
        batch = buffer.sample(args.batch_size)

        epi_len += len(episode)
        if episode[-1].rewards["blue"] == 1:
            nwins += 1
        
        current_qvals, predicted_qvals = {}, {}
        for agent in training_agents:
            current_q, predicted_q = agent.compute_q(batch)
            current_qvals[agent]   = current_q
            predicted_qvals[agent] = predicted_q
        
        states, _, rewards, _, dones, _, _, _, _ = zip(*batch)
        current_q_tot   = mixer(current_qvals, states)
        predicted_q_tot = target_mixer(predicted_qvals, states)

        # for test
        #current_q_tot = current_qvals[agent]
        #predicted_q_tot = predicted_qvals[agent]
        
        rewards = torch.tensor([reward["blue"] for reward in rewards])
        dones = torch.tensor(dones, dtype=torch.float)
        targets = rewards + args.gamma * (1.-dones) * predicted_q_tot

        models["model"].zero_grad()
        loss = F.mse_loss(current_q_tot, targets.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models["model"].parameters(), args.clip)
        models["model"].optimizer.step()

        if step_idx > 0 and step_idx % args.sync_interval == 0:
            models["target"].load_state_dict(models["model"].state_dict())
            target_mixer.load_state_dict(mixer.state_dict())
        
        if step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
            s  = f"Step {step_idx}: loss: {loss:8.4f} - "
            s += f"Average length: {epi_len/PRINT_INTERVAL:5.2f} - "
            s += f"win ratio: {nwins/PRINT_INTERVAL:4.3f} - "
            #s += f"epsilon: {agent.scheduler():4.3f} - "
            print(s)
            epi_len, nwins = 0, 0
        
        ex.log_scalar(f'length', len(episode), step=step_idx+1)
        ex.log_scalar(f'win', int(episode[-1].rewards["blue"] == 1), step=step_idx+1)
        ex.log_scalar(f'loss', loss.item(), step=step_idx+1)

    for agent in training_agents:
        agent.save(args.path+f'RUN_{get_run_id()}_AGENT{agent.id}.p')
    torch.save(models["model"].state_dict(), args.path+f'RUN_{get_run_id()}.torch')
    torch.save(mixer.state_dict(), args.path+f'RUN_{get_run_id()}_MIXER.torch')

#---------------------------------- test -------------------------------------

PRINT_INTERVAL = 5
RENDER = False

@ex.capture
def get_run_id(_run):
    return _run._id

@ex.config
def cgf():
    args = args
    args_dict = dict([(key, Args.__dict__[key]) for key in Args.__dict__ if key[0] != '_'])


@ex.automain
def run(args):
    train(args)
    
"""
if __name__ == '__main__':
    run()
"""