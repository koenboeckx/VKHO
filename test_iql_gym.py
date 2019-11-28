import game
from game import agents
from game.gym_envs import Environment
from marl import iql, agent_models
from game.envs import unflatten, State
import torch

import argparse
from datetime import datetime

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_iql_gym')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    mini_batch_size = 8
    buffer_size = 200 # 10000
    sync_rate = 5000 # 10000
    n_steps = 25000
    lr = 0.1
    gamma = 0.99
    n_hidden = 128

@ex.automain
def run(mini_batch_size, buffer_size, sync_rate,
         n_steps, lr, gamma, n_hidden):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = iql.GymAgent(0, device)

    agents = [agent]
    env = Environment(agents)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create and initialize model for agent
    agent.set_model(env.state_space[0], env.n_actions,
                        n_hidden=n_hidden, lr=lr, device=device)

    iql.train(env, agents,
            mini_batch_size=mini_batch_size,
            buffer_size=buffer_size,
            sync_rate=sync_rate,
            n_steps=n_steps, lr=lr,
            gamma=gamma, experiment=ex)
