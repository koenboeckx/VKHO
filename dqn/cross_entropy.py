"""
The CROSS-ENTROPY method:
1. Play N episodes using current model and environment
2. Calculate the total reward for every episode and decide
    on a reward boundary. Usually, we use some percentile of
    all rewards, suth as 50th or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observaitons
    as the input and issued actions as the desired outputs.
5. Repeat from step 1 until convergence.
"""
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch import optim
import gym

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('cross_entropy')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    hidden_size = 128
    batch_size  = 16
    percentile  = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=[
    'observation', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs,
                                         action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward,
                                 steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = [s.reward for s in batch]
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs, train_act = [], []
    for example in batch:
        if example.reward < reward_bound:
            continue                                            
        train_obs.extend([step.observation for step in example.steps])
        train_act.extend([step.action for step in example.steps])
    
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

@ex.automain
def run(hidden_size, batch_size, percentile):
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, hidden_size, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(iterate_batches(env, net, batch_size)):
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, percentile)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, act_v)
        loss_v.backward()
        optimizer.step()

        print("{}: loss = {:.3f}, reward = {:.1f}, reward_bound={:.1f}".format(
            iter_no, loss_v.item(), reward_m, reward_b
        ))

        ex.log_scalar('loss', loss_v.item())
        ex.log_scalar('reward_bound', reward_b)
        ex.log_scalar('reward_mean', reward_m)

        if reward_m > 199:
            print('Solved!')
            break