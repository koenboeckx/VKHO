import game
from game.envs import Environment
from marl import iql, agent_models
from game.envs import unflatten, State
import torch

import argparse

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_iql')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    mini_batch_size = 256
    buffer_size = 10000
    sync_rate = 10000
    n_steps = 25000
    lr = 0.01
    board_size = 5
    gamma = 0.99

@ex.automain
def run(mini_batch_size, buffer_size, sync_rate,
         n_steps, lr, board_size, gamma):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent0 = iql.IQLAgent(0, device=device, board_size=board_size)
    agent1 = iql.IQLAgent(1, device=device, board_size=board_size)

    agent_list = [
        agent0, # Team 1
        agent1, # Team 1
        game.agents.RandomTank(2), # Team 2
        game.agents.RandomTank(3)  # Team 2
    ]

    agents = [agent0, agent1]
    env = Environment(agent_list, size=board_size)

    # create and initialize model for agent
    input_shape = (1, env.board_size, env.board_size)
    for agent in agents:
        agent.set_model(input_shape, env.n_actions, lr)

    iql.train(env, [agent0, agent1],
                mini_batch_size=mini_batch_size,
                buffer_size=buffer_size,
                sync_rate=sync_rate,
                n_steps=n_steps, lr=lr,
                gamma=gamma, experiment=ex)
    TEST = False
    if TEST:
        filenames = ['./marl/models/iql_agent_0_1241.torch']
                    #'./marl/models/iql_agent_1_1779.torch']
        iql.test(env, [agent0], filenames)
        print('ok')


