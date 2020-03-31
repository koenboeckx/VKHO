import random
from collections import namedtuple

import settings

Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations',
    'hidden', 'next_obs', 'unavailable_actions'
])

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
    
def generate_episode(env, render=False):
    episode = []
    state, done = env.reset(), False
    observations = env.get_all_observations()
    n_steps = 0

    for agent in env.agents:        # for agents where it matters,
        agent.set_hidden_state()    # set the init hidden state of the RNN

    while not done:
        unavailable_actions = env.get_unavailable_actions()
        
        # keep record of hidden state of the agents to store in experience
        hidden = {}
        for agent in env.agents:
            hidden[agent] = agent.get_hidden_state()
        
        actions = env.act(observations)

        if render:
            print(f"Step {n_steps}")
            env.render()
            print([action.name for action in actions.values()])

        next_state, rewards, done, _ = env.step(actions)
        next_obs = env.get_all_observations()
        
        # episodes that take long are not allowed and penalized for both agents
        n_steps += 1
        if n_steps > settings.Args.max_episode_length:
            done = True
            rewards = {'blue': -1, 'red': -1}

        episode.append(Experience(state, actions, rewards, next_state, done, observations, hidden, next_obs, unavailable_actions))
        state = next_state
        observations = next_obs
    
    if render:
        print(f"Game won by team {env.terminal(next_state)}")
    return episode

def get_args(config):
    "Generate an args Class based on the entries from the yaml config file"
    class Args:
        def __init__(self):
            for key in config:
                setattr(self, key, config[key])
            self.n_agents = self.n_enemies + self.n_friends
            self.n_actions = 6 + self.n_enemies # 6 fixed actions + 1 aim action per enemy
            self.n_inputs  = 4 + 3*(self.n_friends - 1) + 3*self.n_enemies + self.n_enemies# see process function in models.py
    return Args()

if __name__ == '__main__':
    from env import Environment, Agent
    from settings import args
    team_blue = [Agent(idx, "blue") for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)
    episode = generate_episode(env, render=True)