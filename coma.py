"""[18Mar20]
Counterfactual Multi-Agent Policy Gradients
"""

import copy
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from env import *
from utilities import  get_args
from models import IACModel, IACRNNModel
from settings import *

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'IAC')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))
ex.add_config('default_config.yaml')    # requires PyYAML   

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
                idx += 3
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


class ComaActor(nn.Module): # TODO: add last action as input
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.rnn_hidden_dim = args.n_hidden
        self.fc1 = nn.Linear(input_shape, args.n_hidden) 
        self.rnn = nn.GRUCell(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.n_hidden)

        self.policy = nn.Linear(args.n_hidden, n_actions)
        self.value  = nn.Linear(args.n_hidden, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        logits = self.policy(q)
        value  = self.value(q)
        return value, logits, h

class ComaCritic(nn.Module):
    def __init__(self):
        pass

class COMAAgent(Agent):
    def __init__(self, id, team):
        super().__init__(id, team)
        
    def set_models(self, models):
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def act(self, obs):
        unavailable_actions = self.env.get_unavailable_actions()[self]
        with torch.no_grad():
            _, logits, self.hidden_state = self.model(obs.unsqueeze(0), self.hidden_state)
            
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

class MultiAgentController:
    def __init__(self, env, agents, models):
        self.env = env # check needed here?
        self.agents = agents
        self.model  = models["model"]
        self.target = models["target"]
        self.parameters = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr)
    
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
        
        observations = observations.reshape(batch_size * len(self.agents), -1)
        next_obs = next_obs.reshape(batch_size * len(self.agents), -1)
        hidden = hidden.reshape(batch_size * len(self.agents), -1)
        next_hidden = next_hidden.reshape(batch_size * len(self.agents), -1)

        current_vals,  logits, _ = self.model(observations, hidden)
        predicted_vals, _,     _ = self.target(next_obs, next_hidden)

        current_vals = current_vals.reshape(batch_size, len(self.agents), -1)
        predicted_vals = predicted_vals.reshape(batch_size, len(self.agents), -1)

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

def train():
    team_blue = [IACAgent(idx, "blue") for idx in range(args.n_friends)]
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents, args)

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

    from os.path import expanduser
    home = expanduser("~")
    #for agent in training_agents:
    #    agent.save(home+args.path+f'RUN_{get_run_id()}_AGENT{agent.id}.p')
    torch.save(models["model"].state_dict(), home+args.path+f'RUN_{get_run_id()}_MODEL.torch')
    

# -------------------------------------------------------------------------------------
PRINT_INTERVAL = 100

@ex.capture
def get_run_id(_run):
    return _run._id
    #print(_run.experiment_info["name"])

@ex.automain
def run(_config):
    global args
    args = get_args(_config)
    train()

"""
if __name__ == '__main__':
    run(args)
"""    