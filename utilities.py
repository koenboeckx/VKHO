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
    
def generate_episode(env, args, render=False):
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
        if n_steps > args.max_episode_length:
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

def train_iteratively(args):
    # iteration 1
    team_blue = [PGAgent(idx, "blue") for idx in range(args.n_friends)]
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)

    args.n_actions = 6 + args.n_enemies
    args.n_inputs  = 4 + 3*(args.n_friends-1) + 3*args.n_enemies
    
    model = ForwardModel(input_shape=args.n_inputs, n_actions=args.n_actions)
    
    for agent in training_agents:
        agent.set_model(model)

    training_agents = train_agents(env, training_agents, args)
    trained_model = copy.deepcopy(training_agents[0].model)
    
    for iteration in range(args.n_iterations):
        args.n_steps = 10000 * (iteration + 2)
        team_blue = [PGAgent(idx, "blue") for idx in range(args.n_friends)]
        team_red  = [PGAgent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

        training_agents = team_blue

        agents = team_blue + team_red
        env = Environment(agents, args)

        model = ForwardModel(input_shape=args.n_inputs, n_actions=args.n_actions)
        model.load_state_dict(trained_model.state_dict())
        model.eval()
        for agent in team_red:
            agent.set_model(model)
        
        model = ForwardModel(input_shape=args.n_inputs, n_actions=args.n_actions)
        for agent in team_blue:
            agent.set_model(model)
        
        training_agents = train_agents(env, training_agents, args)
        trained_model = copy.deepcopy(training_agents[0].model)
    torch.save(trained_model.state_dict(), args.path+f'RUN_{get_run_id()}.torch')

def train_agents(env, training_agents, args):
    epi_len, nwins = 0, 0
    n_episodes = 0
    for step_idx in range(int(args.n_steps/args.n_episodes_per_step)):
        batch = []
        for _ in range(args.n_episodes_per_step):
            episode = generate_episode(env, args)
            n_episodes += 1 
            batch.extend(episode)

            epi_len += len(episode)
            reward = episode[-1].rewards["blue"]

            ex.log_scalar('length', len(episode))
            ex.log_scalar('reward', reward)
            ex.log_scalar(f'win_blue', int(episode[-1].rewards["blue"] == 1))
            ex.log_scalar(f'win_red', int(episode[-1].rewards["red"] == 1))

            if episode[-1].rewards["blue"] == 1:
                nwins += 1

        for agent in training_agents:
            loss = agent.update(batch)
            ex.log_scalar(f'loss{agent.id}', loss)

        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/args.n_episodes_per_step:5.2f} - "
        s += f"win ratio: {nwins/args.n_episodes_per_step:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0

    return training_agents

if __name__ == '__main__':
    from env import Environment, Agent
    from settings import args
    team_blue = [Agent(idx, "blue") for idx in range(args.n_friends)] 
    team_red  = [Agent(idx + args.n_friends, "red") for idx in range(args.n_enemies)] 

    training_agents = team_blue

    agents = team_blue + team_red
    env = Environment(agents)
    episode = generate_episode(env, render=True)