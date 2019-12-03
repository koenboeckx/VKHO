"""Tabular Q-learning applied to FrozenLake-v0"""

import gym
import collections
import random


from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_frozen_lake')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))
                                

@ex.config
def cfg():
    n_steps = 1000000
    n_test_episodes = 10
    gamma = 0.9
    alpha = 0.2

def select_action(values, epsilon=0.0):
    if random.random() < epsilon:
        return random.randrange(0, len(values))
    else:
        _, action = max([(val, action) for action, val in enumerate(values)])
        return action

def play_episode(Q):
    test_env = gym.make('FrozenLake-v0')
    total_reward = 0.0
    state = test_env.reset()
    done = False
    while not done:
        action = select_action(Q[state])
        next_state, reward, done, _ = test_env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def average(list_):
    return sum(list_)/len(list_)

@ex.automain
def q_learning(n_steps, n_test_episodes, gamma, alpha):
    env = gym.make('FrozenLake-v0')
    n_actions = env.action_space.n
    #n_states  = env.observation_space.n  
    Q = collections.defaultdict(lambda: [0,]*n_actions)

    state = env.reset()
    for step_idx in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        delta = reward + gamma * max(Q[next_state]) - Q[state][action]
        if done:
            print('{}: episode done with reward {}'.format(step_idx, reward))
        Q[state][action] += alpha * delta
        state = env.reset() if done else next_state

        avg_reward = average([play_episode(Q) for _ in range(n_test_episodes)])
        ex.log_scalar('avg_reward', avg_reward)
        ex.log_scalar('step_idx', step_idx)