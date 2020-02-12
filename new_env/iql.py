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
            raise ValueError((f"x should be (list of) State or Observation"))
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
    def __init__(self, id, models, params):
        super().__init__(id, params)
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()
        self.scheduler = params['scheduler'](start=1.0, stop=0.1, steps=params['scheduler_steps'])
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def act(self, obs):
        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in range(self.env.n_actions)
                            if action not in unavail_actions]
        eps = self.scheduler()
        if random.random() < eps:
            return random.choice(avail_actions)
        else:
            with torch.no_grad():
                qvals = self.model([obs])
                # remove unavailable actions
                for action in unavail_actions:
                    qvals[0][action] = -np.infty
                action = qvals.max(1)[1].item()
                return action
    
    def update(self, batch):
        _, actions, rewards, _, dones, observations, next_obs, unavailable_actions = zip(*batch)
        actions = [action[self] for action in actions]
        rewards = torch.tensor([reward[self] for reward in rewards])
        observations = [obs[self] for obs in observations]
        next_obs = [obs[self] for obs in next_obs]
        dones = torch.tensor(dones, dtype=torch.float)
        
        current_qvals = self.model(observations)[range(len(batch)), actions]
        
        predicted_qvals = self.target(next_obs)
        # Set unavailable action to very low Q !!
        for idx in range(len(unavailable_actions)):
            unavail_actions = unavailable_actions[idx][self]
            predicted_qvals[idx][unavail_actions] = -np.infty

        predicted_qvals_max = predicted_qvals.max(1)[0]
        targets = rewards + params['gamma']*(1.-dones) * predicted_qvals_max

        self.model.optimizer.zero_grad()
        loss = F.mse_loss(current_qvals, targets.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), params['clip'])
        self.model.optimizer.step()
        return loss.item()


Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations', 'next_obs', 'unavailable_actions'
])


def generate_episode(env, render=False):
    episode = []
    state, done = env.reset(), False
    observations = dict([(agent, env.get_observation(agent)) for agent in env.agents])
    n_steps = 0
    while not done:
        unavailable_actions = env.get_unavailable_actions()
        actions = dict([(agent, agent.act(observations[agent])) for agent in env.agents])

        if render:
            print(f"Step {n_steps}")
            env.render()
            print([all_actions[actions[agent]] for agent in env.agents])

        next_state, rewards, done, _ = env.step(actions)
        next_obs = dict([(agent, env.get_observation(agent)) for agent in env.agents])
        
        # episodes that take long are not allowed and penalized for both agents
        n_steps += 1
        if n_steps > params['max_episode_length']:
            done = True
            rewards = {env.agents[0]: -1,
                       env.agents[1]: -1}

        episode.append(Experience(state, actions, rewards, next_state, done, observations, next_obs, unavailable_actions))
        state = next_state.copy()
        observations = next_obs.copy()

    return episode

def generate_models():
    model  = QLModel(input_shape=6, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    target = QLModel(input_shape=6, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    return {"model": model, "target": target}


#---------------------------------- test -------------------------------------
params = {
    'board_size':           5,
    'init_ammo':            5,
    'max_range':            5,
    'step_penalty':         0.01,
    'max_episode_length':   100,
    'gamma':                0.9,
    'n_hidden':             128,
    'scheduler':            LinearScheduler,
    'buffer_size':          5000,
    'batch_size':           512,
    'n_steps':              50000,
    'sync_interval':        20,
    'lr':                   0.1,
    'clip':                 10,
    'scheduler_steps':      100000,
}
PRINT_INTERVAL = 100
RENDER = False

def test_generate_episode():
    agents = [Agent(0, params), Agent(1, params)]
    env = SimpleEnvironment(agents)
    episode = generate_episode(env)
    print(len(episode))

def test_take_action():
    models = generate_models()
    agents = [QLAgent(0, models, params), Agent(1, params)]
    training_agents = [agents[0]]
    env = SimpleEnvironment(agents, params)
    buffer = ReplayBuffer(size=params['buffer_size'])
    epi_len, nwins = 0, 0
    for step_idx in range(params['n_steps']):
        episode = generate_episode(env)
        buffer.insert_list(episode)
        if not buffer.can_sample(params['batch_size']):
            continue
        epi_len += len(episode)
        if episode[-1].rewards[agents[0]] == 1:
            nwins += 1
        batch = buffer.sample(params['batch_size'])
        for agent in training_agents:
            loss = agent.update(batch)
            if step_idx > 0 and step_idx % params['sync_interval'] == 0:
                agent.sync_models()
    
            if step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
                s  = f"Step {step_idx}: loss for agent {agent}: {loss:5.4f} - "
                s += f"Average length: {epi_len/PRINT_INTERVAL:5.2f} - "
                s += f"win ratio: {nwins/PRINT_INTERVAL:4.3f} - "
                s += f"epsilon: {agent.scheduler():4.3f} - "
                print(s)
                epi_len, nwins = 0, 0
                #_ = generate_episode(env, render=True)
        

if __name__ == '__main__':
    test_take_action()