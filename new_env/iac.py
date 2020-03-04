"""[23Feb20]
Independent Actor-Critic
Uses independent A2C agents with common weights
"""

import copy
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import Experience, generate_episode, get_args
from models import IACModel, IACRNNModel
from settings import *

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'IAC-{args.n_friends}v{args.n_enemies}')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))
ex.add_config('new_env/default_config.yaml')    # requires PyYAML
args = get_args(ex)                                 

class IACAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        
    def set_models(self, models):
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def act(self, obs):
        if not obs.alive: # if not alive, do nothing
            return Action(0, 'do_nothing', target=None)
        
        unavailable_actions = self.env.get_unavailable_actions()[self]
        with torch.no_grad():
            if args.model == 'RNN':
                _, logits, self.hidden_state = self.model([obs], self.hidden_state)
            else:
                _, logits = self.model([obs])
            
            for action in unavailable_actions:
                logits[0][action.id] = -np.infty

            action_idx = Categorical(logits=logits[0]).sample().item()
        return self.actions[action_idx]
    
    def compute_returns(self, batch):
        _, _, rewards, _, dones, _, _, _, _ = zip(*batch)
        rewards = [reward[self.team] for reward in rewards]
        returns, R = [], 0.0
        for reward, done in reversed(list(zip(rewards, dones))):
            if done:
                R = 0.0
            R = reward + args.gamma * R
            returns.insert(0, R)
        return returns

    def set_hidden_state(self):
        self.hidden_state = self.model.init_hidden()

    def update(self, batch):
        _, actions, rewards, _, dones, observations, hidden, next_obs, unavail = zip(*batch)
        
        # only perform updates on actions performed while alive
        self_alive_idx = [idx for idx, obs in enumerate(observations) if obs[self].alive]
        observations = [obs[self] for obs in observations]
        next_obs     = [obs[self] for obs in next_obs]
        dones = torch.tensor(dones, dtype=torch.float)
        hidden = [hidden_state[self] for hidden_state in hidden]

        rewards = torch.tensor([reward[self.team] for reward in rewards])
        returns = torch.tensor(self.compute_returns(batch))[self_alive_idx]
        
        if args.model == 'RNN':
            logits = torch.zeros(len(batch), args.n_actions)
            values = torch.zeros(len(batch), 1)
            next_vals = torch.zeros(len(batch), 1)
            for t in range(len(batch)):
                values[t, :], logits[t, :], h = self.model([observations[t]], hidden[t]) # TODO: since batch is sequence of episodes, is this the best use of hidden state?
                next_vals[t, :], _, _ = self.target([next_obs[t]], h)
        else:
            values, logits = self.model(observations)        

        actions = torch.tensor([action[self].id for action in actions])[self_alive_idx]

        for idx in self_alive_idx:
            unavail_actions = unavail[idx][self]
            unavail_ids = [action.id for action in unavail_actions]
            logits[idx][unavail_ids] = -99999
        
        target = rewards + args.gamma * next_vals.squeeze() * (1.0 - dones)
        advantage = target.detach() - values.squeeze() # TODO: detach  ok?
        advantage = advantage[self_alive_idx]        

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob_act = log_prob[self_alive_idx, actions]
        log_prob_act_val = advantage.detach() * log_prob_act # TODO: where to detach ?
        #log_prob_act_val = returns * log_prob_act

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * log_prob).sum(dim=1)
        loss_entropy = entropy.mean()
        
        loss_pol = -log_prob_act_val.mean()
        loss_val = advantage.pow(2).mean()
        loss = loss_pol + loss_val - args.beta * loss_entropy # try to maximize entropy

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])

        return {'loss':         loss.item(),
                'policy_loss':  loss_pol.item(),
                'value_loss':   loss_val.item(),
                'entropy':      loss_entropy.item(),
                'grads_l2':     np.sqrt(np.mean(np.square(grads))),
                'grads_var':    np.var(grads),
        }

def play_from_file(filename):
    model = IACModel(input_shape=13, n_actions=7)
    model.load_state_dict(torch.load(filename))

    team_red  = [IACAgent(2, "red"), PGAgent(3, "red")]
    team_blue = [Agent(0, "blue"),  Agent(1, "blue")]

    agents = team_blue + team_red

    env = Environment(agents)
    _ = generate_episode(env, render=True)

def generate_model(input_shape, n_actions):
    if args.model == 'RNN':
        model  = IACRNNModel(input_shape=input_shape, n_actions=n_actions)
        target = IACRNNModel(input_shape=input_shape, n_actions=n_actions)
    else:
        model  = IACModel(input_shape=input_shape, n_actions=n_actions)
        target = IACModel(input_shape=input_shape, n_actions=n_actions)
    return {"model": model, "target": target}

def train(args):
    team_blue = [IACAgent(idx, "blue") for idx in range(args.n_friends)]
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)

    args.n_actions = 6 + args.n_enemies
    args.n_inputs  = 4 + 3*(args.n_friends-1) + 3*args.n_enemies
    
    # setup model   
    models = generate_model(input_shape=args.n_inputs, n_actions=args.n_actions)
    
    for agent in training_agents:
        agent.set_models(models)

    epi_len, nwins = 0, 0
    n_episodes = 0
    ex.log_scalar(f'win', 0.0, step=0) # forces start of run at 0 wins ()
    for step_idx in range(int(args.n_steps/args.n_episodes_per_step)):
        batch = []
        for _ in range(args.n_episodes_per_step):
            episode = generate_episode(env)
            n_episodes += 1 
            batch.extend(episode)

            epi_len += len(episode)
            reward = episode[-1].rewards["blue"]

            ex.log_scalar('length', len(episode), step=n_episodes)
            ex.log_scalar('reward', reward, step=n_episodes)
            ex.log_scalar(f'win', int(episode[-1].rewards["blue"] == 1), step=n_episodes + 1)

            if episode[-1].rewards["blue"] == 1:
                nwins += 1

        for agent in training_agents:
            stats = agent.update(batch)
            ex.log_scalar(f'policy_loss{agent.id}', stats['policy_loss'], step=n_episodes)
            ex.log_scalar(f'value_loss{agent.id}',  stats['value_loss'],  step=n_episodes)
            ex.log_scalar(f'loss{agent.id}', stats['loss'], step=n_episodes)
            ex.log_scalar(f'entropy{agent.id}', stats['entropy'], step=n_episodes)
            ex.log_scalar(f'grads{agent.id}', stats["grads_l2"], step=n_episodes)
            ex.log_scalar(f'grads_var{agent.id}', stats["grads_var"], step=n_episodes)

            if step_idx % 50 == 0: #args.sync_interval == 0:
                agent.sync_models()
                print(f'sync at {step_idx * args.n_episodes_per_step}')

        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/args.n_episodes_per_step:5.2f} - "
        s += f"win ratio: {nwins/args.n_episodes_per_step:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0

        #_ = generate_episode(env, render=True)


    path = '/home/koen/Programming/VKHO/new_env/agent_dumps/'
    for agent in training_agents:
        agent.save(path+f'RUN_{get_run_id()}_AGENT{agent.id}.p')
    torch.save(models["model"].state_dict(), path+f'RUN_{get_run_id()}.torch')


# -------------------------------------------------------------------------------------
PRINT_INTERVAL = 100

@ex.capture
def get_run_id(_run):
    return _run._id
    #print(_run.experiment_info["name"])

@ex.automain
def run():
    train(args)

"""
if __name__ == '__main__':
    run(args)
"""    