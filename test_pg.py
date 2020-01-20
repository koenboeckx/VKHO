import game
from game.envs import Environment
from marl import pg, agent_models
from game.envs import unflatten, State
import torch

import argparse
from datetime import datetime

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('policy_gradients')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    rl_type = 'reinforce' # 'reinforce' or 'actor-critic'
    n_hidden = 128
    lr = 0.001
    n_episodes = 20
    n_steps = 1000
    board_size = 7
    agent_type = 'normal' # 'normal or 'gru'
    hidden_size = 512 # size of hidden vector in RNN

@ex.automain
def run(rl_type, n_hidden, lr, n_episodes, n_steps, board_size, agent_type, hidden_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if agent_type == 'normal':
        agent0 = pg.PGAgent(0, device, board_size=board_size)
        agent1 = pg.PGAgent(1, device, board_size=board_size)
    elif agent_type == 'gru':
        agent0 = pg.PG_GRUAgent(0, device, board_size=board_size)
        agent1 = pg.PG_GRUAgent(1, device, board_size=board_size)

    agent_list = [
        agent0, # Team 1
        agent1, # Team 1
        #agents.RandomTank(1), # Team 1
        game.agents.RandomTank(2), # Team 2
        game.agents.RandomTank(3)  # Team 2
    ]
    agents = [agent0, agent1]
    env = Environment(agent_list, size=board_size)
         
    for agent in agents:
        if agent_type == 'normal':
            agent.set_model((1, env.board_size, env.board_size), env.n_actions,
                            lr=lr)
        elif agent_type == 'gru':
            agent.set_model((1, env.board_size, env.board_size), env.n_actions,
                            lr=lr, hidden_size=hidden_size)
        print(agent)
        print(agent.model)
    
    if rl_type == 'reinforce':
        pg.reinforce(env, agents, n_episodes=n_episodes,
                    n_steps=n_steps, experiment=ex)
    elif rl_type == 'actor-critic':
        pg.actor_critic(env, agents, n_episodes=n_episodes,
                    n_steps=n_steps, experiment=ex)



    filenames = ['/home/koen/Programming/VKHO/marl/models/pg_agent0_20191113_103251.torch',
                '/home/koen/Programming/VKHO/marl/models/pg_agent1_20191113_103251.torch', 
    ]
    #pg.test_agents(env, agents, filenames)