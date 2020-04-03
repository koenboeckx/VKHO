"""Policy Gradients Algorithm"""

from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import get_args
from new_utilities import generate_episode, Experience

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'PG2')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))
ex.add_config('default_config.yaml')    # requires PyYAML 

from profilehooks import profile

class RNNModel(nn.Module): # TODO: add last action as input
    def __init__(self, input_shape, n_actions, args):
        super().__init__()
        self.rnn_hidden_dim = args.n_hidden
        self.fc1 = nn.Linear(input_shape, args.n_hidden) 
        self.rnn = nn.GRUCell(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class PGAgent(Agent):
    def __init__(self, id, team, args):
        super().__init__(id, team)
        self.args = args

    def set_model(self, model):
        self.model = model
    
    def act(self, obs, test_mode):
        #if obs[2] == 0: # player is dead, thus do_nothing
        #    return self.actions[0]

        unavail_actions = self.env.get_unavailable_actions()[self]
                        
        with torch.no_grad():
            logits, self.hidden_state = self.model(obs.unsqueeze(0), self.hidden_state)
            logits = logits[0]
            for action in unavail_actions:
                logits[action.id] = -np.infty
            action_idx = Categorical(logits=logits).sample().item()
        return self.actions[action_idx]

    def set_hidden_state(self):
        self.hidden_state = self.model.init_hidden()

    def _build_inputs(self, batch):
        states, actions, rewards, next_states, dones, observations,\
            hidden, next_obs, next_hidden, unavailable_actions = zip(*batch)
        
        # transform all into format we require
        states       = torch.stack(states)
        next_states  = torch.stack(next_states)

        observations = torch.stack(observations)[:, self.id, :]
        next_obs     = torch.stack(next_obs)[:, self.id, :]
        hidden       = torch.stack(hidden).squeeze()[:, self.id, :]
        next_hidden  = torch.stack(next_hidden).squeeze()[:, self.id, :]
        actions      = torch.stack(actions)[:, self.id]
        rewards      = torch.tensor([reward['blue'] for reward in rewards]).unsqueeze(-1)
        dones        = torch.tensor(dones, dtype=torch.float).unsqueeze(-1)
        unavail      = torch.stack(unavailable_actions)[:, self.id, :]
        return states, next_states, observations, next_obs, hidden, next_hidden,\
             actions, rewards, dones, unavail

    def compute_returns(self, batch):
        _, _, rewards, _, dones, _, _, _, _, _ = zip(*batch)
        rewards = [reward[self.team] for reward in rewards]
        returns, R = [], 0.0
        for reward, done in reversed(list(zip(rewards, dones))):
            if done:
                R = 0.0
            R = reward + self.args.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, batch):
        _, _, observations, _, hidden, _, actions,\
            _, _, unavail = self._build_inputs(batch)

        logits, _ = self.model(observations, hidden)
        # set unavailable actions to a very low value
        logits[unavail == 1.] = -999

        # only perform updates on actions performed while alive
        alive_idx = observations[:, 2] == 1.
        actions = actions[alive_idx]

        expected_returns =  torch.tensor(self.compute_returns(batch))[alive_idx]
        
        # set unavailable actions to a very low value
        logits[unavail == 1.] = -999

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob_act = log_prob[alive_idx, actions]
        log_prob_act_val = (expected_returns - expected_returns.mean()) * log_prob_act
        loss = -log_prob_act_val.mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])
        
        return {'loss':         loss.item(),
                'grads_l2':     np.sqrt(np.mean(np.square(grads))),
                'grads_var':    np.var(grads),
        }

def train(args):
    team_blue = [PGAgent(idx, "blue", args) for idx in range(args.n_friends)]
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    if args.env_type == 'normal':
        env = Environment(agents, args)
    elif args.env_type == 'restricted':
        env = RestrictedEnvironment(agents, args)

    args.n_actions = 6 + args.n_enemies
    args.n_inputs  = 4 + 3*(args.n_friends-1) + 3*args.n_enemies + args.n_enemies
    
    # setup model
    model = RNNModel(input_shape=args.n_inputs, n_actions=args.n_actions, args=args)
    
    for agent in training_agents:
        agent.set_model(model)

    epi_len, nwins = 0, 0
    n_episodes = 0
    ex.log_scalar(f'win', 0.0, step=n_episodes + 1) # forces start of run at 0 wins ()
    for step_idx in range(int(args.n_steps/args.n_episodes_per_step)):
        batch = []
        for _ in range(args.n_episodes_per_step):
            episode = generate_episode(env, args)
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
            ex.log_scalar(f'loss{agent.id}', stats["loss"], step=n_episodes)
            ex.log_scalar(f'grads{agent.id}', stats["grads_l2"], step=n_episodes)
            ex.log_scalar(f'grads_var{agent.id}', stats["grads_var"], step=n_episodes)


        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/args.n_episodes_per_step:5.2f} - "
        s += f"win ratio: {nwins/args.n_episodes_per_step:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0

        #_ = generate_episode(env, render=True)

    from os.path import expanduser
    home = expanduser("~")
    #for agent in training_agents:
    #    agent.save(home+args.path+f'RUN_{get_run_id()}_AGENT{agent.id}.p')
    torch.save(model.state_dict(), home+args.path+f'RUN_{get_run_id()}.torch')

# -------------------------------------------------------------------------------------
PRINT_INTERVAL = 100

@ex.capture
def get_run_id(_run): # enables saving model with run id
    return _run._id

@ex.automain
#@profile
def run(_config):
    global args
    args = get_args(_config)
    train(args)

"""
if __name__ == '__main__':
    import yaml
    from utilities import get_args
    args = get_args(yaml.load(open('default_config.yaml', 'r')))
    train(args)
"""