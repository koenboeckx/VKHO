from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import Experience, generate_episode
from models import ForwardModel, RNNModel
from settings import *

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('PG-2v2')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

class PGAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        
    def set_model(self, model):
        self.model = model
    
    def act(self, obs):
        if not obs.alive: # if not alive, do nothing
            return Action(0, 'do_nothing', target=None)
        unavail_actions = self.env.get_unavailable_actions()[self]
        with torch.no_grad():
            logits, self.hidden_state = self.model([obs], self.hidden_state)
            for action in unavail_actions:
                logits[0][action.id] = -np.infty
            action_idx = Categorical(logits=logits).sample().item()
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
        _, actions, rewards, _, _, observations, hidden, _, unavail = zip(*batch)
        
        # only perform updates on actions performed while alive
        self_alive_idx = [idx for idx, obs in enumerate(observations) if obs[self].alive]
        observations = [obs[self] for obs in observations]

        rewards = [reward[self.team] for reward in rewards]
        expected_returns = torch.tensor(self.compute_returns(batch))[self_alive_idx]
        actions = torch.tensor([action[self].id for action in actions])[self_alive_idx]
        hidden = [hidden_state[self] for hidden_state in hidden]

        logits = torch.zeros(len(batch), args.n_actions)
        for t in range(len(batch)):
            logits[t, :], _ = self.model([observations[t]], hidden[t]) # TODO: since batch is sequence of episodes, is this the best use of hidden state?

        for idx in self_alive_idx:
            unavail_actions = unavail[idx][self]
            unavail_ids = [action.id for action in unavail_actions]
            logits[idx][unavail_ids] = -999

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
    model = ForwardModel(input_shape=13, n_actions=7)
    model.load_state_dict(torch.load(filename))

    team_red  = [PGAgent(2, "red", model), PGAgent(3, "red", model)]
    team_blue = [Agent(0, "blue"),  Agent(1, "blue")]

    agents = team_blue + team_red

    env = Environment(agents)
    _ = generate_episode(env, render=True)

def train(args):
    team_blue = [PGAgent(idx, "blue") for idx in range(args.n_friends + 1)]
    team_red  = [Agent(2 + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)

    args.n_actions = 6 + args.n_enemies
    args.n_inputs  = 4 + 3*args.n_friends + 3*args.n_enemies
    
    # setup model
    if args.model == 'FORWARD':
        model = ForwardModel(input_shape=args.n_inputs, n_actions=args.n_actions)
    elif args.model == 'RNN':
        model = RNNModel(input_shape=args.n_inputs, n_actions=args.n_actions)
    
    for agent in training_agents:
        agent.set_model(model)

    epi_len, nwins = 0, 0
    n_episodes = 0
    ex.log_scalar(f'win', 0.0, step=n_episodes + 1) # forces start of run at 0 wins ()
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
            loss = agent.update(batch)
            ex.log_scalar(f'loss{agent.id}', loss, step=n_episodes)

        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/args.n_episodes_per_step:5.2f} - "
        s += f"win ratio: {nwins/args.n_episodes_per_step:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0

        #_ = generate_episode(env, render=True)
    
    path = '/home/koen/Programming/VKHO/new_env/agent_dumps/'
    for agent in training_agents:
        agent.save(path+f'RUN_{get_run_id()}_AGENT{agent.id}.p')
    torch.save(model.state_dict(), path+f'RUN_{get_run_id()}.torch')


# -------------------------------------------------------------------------------------
PRINT_INTERVAL = 100

@ex.capture
def get_run_id(_run):
    return _run._id
    #print(_run.experiment_info["name"])

@ex.config
def cgf():
    args = args
    args_dict = dict([(key, Args.__dict__[key]) for key in Args.__dict__ if key[0] != '_'])

@ex.automain
def run(args):
    train(args)

"""
if __name__ == '__main__':
    #filename = '/home/koen/Programming/VKHO/new_env/REINFORCE-2v2_7x7.torch'
    #play_from_file(filename)
    run()
"""