# TODO: use this as test case for sacred

import game
from game import agents
from game.gym_envs import Environment
from marl import pg, agent_models
from game.envs import unflatten, State
import torch

import argparse
from datetime import datetime

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('test_pg')
ex.observers.append(MongoObserver(url='localhost',
                                  db_name='my_database'))

@ex.config
def cfg():
    rl_type = 'reinforce'
    n_hidden = 128
    lr = 0.001
    n_episodes = 10
    n_steps = 1000

@ex.automain
def run(rl_type, n_hidden, lr, n_episodes, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = pg.GymAgent(0, device)

    agents = [agent]
    env = Environment(agents)

    if __name__ == '__main__':
        
        for agent in agents:
            agent.set_model(env.state_space[0], env.n_actions,
                            n_hidden=n_hidden, lr=lr)
        
        if rl_type == 'reinforce':
            pg.reinforce(env, agents, n_episodes=n_episodes,
                        n_steps=n_steps, experiment=ex)
        elif rl_type == 'actor_critic':
            pg.actor_critic2(env, agents, n_episodes=n_episodes,
                        n_steps=n_steps)
