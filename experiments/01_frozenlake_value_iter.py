import gym 
import collections

ENV_NAME = "FrozenLake-v0"

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_{}'.format(ENV_NAME))
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

class Agent:
    def __init__(self, gamma):
        self.gamma = gamma
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards  = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values   = collections.defaultdict(float)
    
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
    
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)] # = dict(s1:c1, s2:c2, ...)
        total = sum(target_counts.values()) # = c1+c2+...
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count/total) * (reward + self.gamma*self.values[tgt_state])
        return action_value
    
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action
    
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)
    
@ex.config
def cfg():
    gamma = 0.9
    test_episodes = 20

@ex.automain
def run(gamma, test_episodes):
    test_env = gym.make(ENV_NAME)
    agent = Agent(gamma)
    
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(test_episodes):
            reward += agent.play_episode(test_env)
        reward /= test_episodes
        ex.log_scalar('reward', reward)
        if reward > best_reward:
            print('Best reward updated {:.3f} -> {:.3f}'.format(best_reward, reward))
            best_reward = reward
        if reward > 0.95:
            print('Solved in {:d} iterations'.format(iter_no))
            break