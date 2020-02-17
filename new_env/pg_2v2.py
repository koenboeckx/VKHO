from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import Experience, generate_episode
from models import ForwardModel
from settings import params

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('PG-2v2')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

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
    
    def compute_returns(self, batch):
        _, _, rewards, _, dones, _, _, _ = zip(*batch)
        rewards = [reward[self] for reward in rewards]
        returns, R = [], 0.0
        for reward, done in reversed(list(zip(rewards, dones))):
            if done:
                R = 0.0
            R = reward + params['gamma'] * R
            returns.insert(0, R)
        return returns

    def update(self, batch):
        _, actions, rewards, _, _, observations, _, unavail = zip(*batch)
        
        # only perform updates on actions performed while alive
        self_alive_idx = [idx for idx, obs in enumerate(observations) if obs[self].alive]
        observations = [obs[self] for obs in observations]

        rewards = [reward[self] for reward in rewards]
        expected_returns = torch.tensor(self.compute_returns(batch))[self_alive_idx]
        actions = torch.tensor([action[self] for action in actions])[self_alive_idx]

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

def play_from_file(filename):
    model = PGModel(input_shape=13, n_hidden=params["n_hidden"],
                    n_actions=len(all_actions), lr=params["lr"])
    model.load_state_dict(torch.load(filename))
    #team_blue = [PGAgent(0, "blue", model, params), PGAgent(1, "blue", model, params)]
    #team_red  = [Agent(2, "red", params),  Agent(3, "red", params)]

    team_red  = [PGAgent(2, "red", model, params), PGAgent(3, "red", model, params)]
    team_blue = [Agent(0, "blue", params),  Agent(1, "blue", params)]

    agents = team_blue + team_red

    env = Environment(agents, params)
    _ = generate_episode(env, render=True)

# -------------------------------------------------------------------------------------

@ex.config
def cfg():
    params = params

PRINT_INTERVAL = 100

@ex.automain
def run(params):
    model = ForwardModel(input_shape=13, n_hidden=params["n_hidden"],
                         n_actions=len(all_actions), lr=params["lr"])

    team_blue = [PGAgent(0, "blue", model, params), PGAgent(1, "blue", model, params)]
    team_red  = [Agent(2, "red", params),  Agent(3, "red", params)]

    training_agents = team_blue
    agents = team_blue + team_red

    env = Environment(agents, params)
    epi_len, nwins = 0, 0
    n_episodes = 0
    ex.log_scalar(f'win', 0.0, step=n_episodes + 1) # forces start of run at 0 wins ()
    for step_idx in range(params["n_steps"]):
        batch = []
        for _ in range(params["n_episodes_per_step"]):
            episode = generate_episode(env)
            n_episodes += 1 
            batch.extend(episode)

            epi_len += len(episode)
            reward = episode[-1].rewards[env.agents[0]]

            ex.log_scalar('length', len(episode), step=n_episodes)
            ex.log_scalar('reward', reward, step=n_episodes)
            ex.log_scalar(f'win', int(episode[-1].rewards[agents[0]] == 1), step=n_episodes + 1)

            if episode[-1].rewards[agents[0]] == 1:
                nwins += 1

        for agent in training_agents:
            loss = agent.update(batch)
            ex.log_scalar(f'loss{agent.id}', loss, step=n_episodes)

        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/params['n_episodes_per_step']:5.2f} - "
        s += f"win ratio: {nwins/params['n_episodes_per_step']:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0
    
    for agent in training_agents:
        agent.save(f'REINFORCE-2v2_7_agent{agent.id}.p.temp')
    torch.save(model.state_dict(), 'REINFORCE-2v2_7x7.torch.temp')

"""
if __name__ == '__main__':
    filename = '/home/koen/Programming/VKHO/new_env/REINFORCE-2v2_7x7.torch'
    play_from_file(filename)
"""