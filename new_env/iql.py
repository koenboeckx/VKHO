from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F

from simple_env import *

class QLModel(nn.Module):   # Q-Learning Model
    def __init__(self, input_shape, n_hidden, n_actions, lr):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
    
    def forward(self, x):
        x = self.process(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        vals = self.fc3(x)
        return vals
    
    def process(self, obs_list):
        "transform list of observations into tensor"
        x = torch.zeros((len(obs_list), 6)) # own pos (2), other pos (2), ammo (1), aim (1)
        if isinstance(obs_list[0], Observation):
            for obs_idx, obs in enumerate(obs_list):
                x[obs_idx, 0:2] = torch.tensor(obs.own_position)
                x[obs_idx, 2:4] = torch.tensor(obs.other_position)
                x[obs_idx, 4]   = obs.ammo / params['init_ammo']
                x[obs_idx, 5]   = int(obs.aim is not None)
        elif isinstance(obs_list[0], State): # intervention to allow also 
            agent = list(obs_list[0].position.keys())[0] # TODO: can't keep working
            obs_list = [Observation(state, agent) for state in obs_list]
            return self.process(obs_list)
        else:
            raise ValueError(f("x should be (list of) State or Observation"))
        return x


class ExponentialScheduler:
    def __init__(self, start, stop, decay=0.99):
        self.stop  = stop
        self.decay = decay
        self.value = start
    
    def __call__(self):
        self.value *= self.value * self.decay
        return max(self.value, self.stop)

class LinearScheduler:
    def __init__(self, start, stop, steps=10000):
        self.start = start
        self.stop  = stop
        self.delta = (start - stop) / steps
        self.t = 0
    
    def __call__(self):
        epsilon =  max(self.start - self.t * self.delta, self.stop)
        self.t += 1
        return epsilon

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.content = []
    
    def __len__(self):
        return len(self.content)
    
    def insert(self, item):
        self.content.append(item)
        if len(self) > self.size:
            self.content.pop(0)
    
    def insert_list(self, items):
        for item in items:
            self.insert(item)
    
    def can_sample(self, N):
        return len(self) >= N
    
    def sample(self, N):
        assert self.can_sample(N)
        return random.sample(self.content, N)

class QLAgent(Agent):
    def __init__(self, id, models):
        super().__init__(id)
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()
        self.scheduler = params['scheduler'](start=1.0, stop=0.1)
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def act(self, obs):
        eps = self.scheduler()
        if random.random() < eps:
            return random.randint(0, self.env.n_actions-1)
        else:
            with torch.no_grad():
                qvals = self.model([obs])
                # remove unavailable actions
                #avail_actions = self.env.get_available_actions(obs)
                #for action in range(len(qvals)):
                #    if action not in avail_actions:
                #        qvals[action] = -np.infty
                action = qvals.max(1)[1].item()
                return action
    
    def update(self, batch):
        states, actions, rewards, next_states, dones, observations = \
            zip(*batch)
        actions = [action[self] for action in actions]
        rewards = torch.tensor([reward[self] for reward in rewards])
        observations = [obs[self] for obs in observations]
        dones = torch.tensor(dones, dtype=torch.float)
        
        current_qvals = self.model(observations)[range(len(batch)), actions]
        predicted_qvals = self.target(next_states).max(1)[0] # TODO: quid next observations?
        targets = rewards + params['gamma']*(1.-dones)*predicted_qvals

        self.model.optimizer.zero_grad()
        loss = F.mse_loss(current_qvals, targets.detach())
        loss.backward()
        self.model.optimizer.step()
        return loss.item()


Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations'
])


def generate_episode(env): # TODO: check max step length
    episode = []
    state, done = env.reset(), False
    while not done:
        observations = dict([(agent, env.get_observation(agent)) for agent in env.agents])
        actions = dict([(agent, agent.act(observations[agent])) for agent in env.agents])
        next_state, rewards, done, _ = env.step(actions)
        episode.append(Experience(state, actions, rewards, next_state, done, observations))
        state = next_state
    return episode

def generate_models():
    model  = QLModel(input_shape=6, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    target = QLModel(input_shape=6, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    return {"model": model, "target": target}


#---------------------------------- test -------------------------------------
params = {
    'board_size':   5,
    'init_ammo':    5,
    'max_range':    3,
    'step_penalty': 0.001,
    'n_hidden':     128,
    'lr':           0.01,
    'scheduler':    LinearScheduler,
    'buffer_size':  500,
    'batch_size':   128,
    'n_steps':      1000,
    'gamma':        0.99,
    'sync_interval':    10,
}

def test_generate_episode():
    agents = [Agent(0), Agent(1)]
    env = SimpleEnvironment(agents)
    episode = generate_episode(env)
    print(len(episode))

def test_take_action():
    models = generate_models()
    agents = [QLAgent(0, models), Agent(1)]
    training_agents = [agents[0]]
    env = SimpleEnvironment(agents)
    buffer = ReplayBuffer(size=params['buffer_size'])
    epi_len = 0
    for step_idx in range(params['n_steps']):
        episode = generate_episode(env)
        epi_len += len(episode)
        buffer.insert_list(episode)
        if not buffer.can_sample(params['batch_size']):
            continue
        batch = buffer.sample(params['batch_size'])
        for agent in training_agents:
            loss = agent.update(batch)
            if step_idx > 0 and step_idx % params['sync_interval'] == 0:
                agent.sync_models()
                print(f"Step {step_idx}: loss for agent {agent}: {loss}")
                print(f"Average length: {epi_len/params['sync_interval']}")
                epi_len = 0
        

if __name__ == '__main__':
    test_take_action()