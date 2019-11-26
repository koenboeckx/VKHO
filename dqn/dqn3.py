"""
Implement tabular Q-learning (discretize states)
"""
import random
import math
from collections import deque, defaultdict

import gym
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('dqn3')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))


degrees = { 1:  0.0174532, # 2*pi/360
            6:  0.1047192,
            12: 0.2094384,
            50: 0.87266}

def discretize_(state):
    """based on http://pages.cs.wisc.edu/~finton/qcontroller.html
    function get_box()"""
    x, x_dot, theta, theta_dot = state
    if x < -.8:  box = 0
    elif x < .8: box = 1
    else:        box = 2

    if x_dot < -.5:   pass
    elif x_dot < 0.5: box += 3
    else:             box += 6

    if theta < -degrees[6]:   pass
    elif theta < -degrees[1]: box += 9
    elif theta < 0:           box += 18
    elif theta < degrees[1]:  box += 27
    elif theta < degrees[6]:  box += 36
    else:                     box += 45

    if theta_dot < -degrees[50]:    pass
    elif theta_dot < degrees[50]:   box += 54
    else:                           box += 108

    return box

def discretize(state, env):
    """from https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578"""
    buckets=(1, 1, 6, 12,) # downscaling factor for each feature
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

class Agent:
    def __init__(self, env, ex=ex, maxlen=100, gamma=0.99, alpha=0.1, decay=0.995):
        self.env = env
        self.ex = ex
        self.n_actions = env.action_space.n
        self.memory = deque(maxlen=maxlen)
        self.Q = defaultdict(lambda: [0,]*self.n_actions) # store the predicted Q-values
        self.gamma = gamma
        
        self.alpha = 1.0
        self.min_alpha = alpha

        self.epsilon = 1.0
        self.decay = decay
        self.epsilon_min = 0.05
     
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            act_vals = self.Q[discretize(state, self.env)]
            action = np.argmax(act_vals)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state, action, reward, next_state, done):
        "update Q based on single sample"
        prediction = self.Q[discretize(state, self.env)][action]
        if not done:
            target = reward + self.gamma * max(self.Q[discretize(next_state, self.env)])
        else:
            target = reward
        diff = target - prediction
        self.Q[discretize(state, self.env)][action] += self.alpha * diff
    
    def replay(self, batchsize):
        "update Q based on sampled minibatch"
        minibatch = random.sample(self.memory, batchsize)
        cum_diff = 0
        for state, action, reward, next_state, done in minibatch:
            prediction = self.Q[discretize(state, self.env)][action]
            if not done:
                target = reward + self.gamma * max(self.Q[discretize(next_state, self.env)])
            else:
                target = reward
            diff = target - prediction
            cum_diff += diff
            self.Q[discretize(state, self.env)][action] += self.alpha * diff
        
        self.ex.log_scalar('avg_diff', cum_diff/len(minibatch))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        if self.alpha > self.min_alpha:
            self.alpha *= self.decay

        self.ex.log_scalar('epsilon', self.epsilon)
        self.ex.log_scalar('alpha', self.alpha)

@ex.config
def cfg():
    n_episodes  = 1000
    batchsize   = 8
    maxlen      = 2000
    gamma       = 0.99
    min_alpha   = 0.1
    decay       = 0.995

#@ex.automain
def run(n_episodes, batchsize, maxlen, gamma, min_alpha, decay):
    "run with minibatch update"
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 499
    agent = Agent(env, ex=ex, maxlen=maxlen, gamma=gamma, alpha=min_alpha, decay=decay)
    for ep_idx in range(n_episodes):
        state = env.reset()
        for time_t in range(500):
            if ep_idx % 500 == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode {}/{}, score: {}'.format(ep_idx, n_episodes, time_t))
                ex.log_scalar('reward', time_t)
                break
        
        if len(agent.memory) > batchsize:
            agent.replay(batchsize)
    env.close()

@ex.automain
def run2(n_episodes, maxlen, gamma, min_alpha, decay):
    "run with update after every step"
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 499
    agent = Agent(env, ex=ex, maxlen=maxlen, gamma=gamma, alpha=min_alpha, decay=decay)
    for ep_idx in range(n_episodes):
        state = env.reset()
        for time_t in range(500):
            if ep_idx % 500 == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            if done:
                print('episode {}/{}, score: {}'.format(ep_idx, n_episodes, time_t))
                ex.log_scalar('reward', time_t)
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.decay
        if agent.alpha > agent.min_alpha:
            agent.alpha *= agent.decay

    env.close()