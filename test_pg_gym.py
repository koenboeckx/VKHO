import game
from game import agents
from game.gym_envs import Environment
from marl import pg, agent_models
from game.envs import unflatten, State
import torch

import argparse
from datetime import datetime

BOARD_SIZE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = pg.GymAgent(0, device)

agents = [agent]
env = Environment(agents, size=BOARD_SIZE)

if __name__ == '__main__':
    
    for agent in agents:
        agent.set_model(env.state_space[0], env.n_actions,
                        lr=0.001)
    
    #pg.reinforce(env, agents, n_episodes=20, n_steps=3000)
    pg.actor_critic2(env, agents, n_steps=1000,
                    n_episodes=10)



    filenames = ['/home/koen/Programming/VKHO/marl/models/pg_agent0_20191113_103251.torch',
                 '/home/koen/Programming/VKHO/marl/models/pg_agent1_20191113_103251.torch', 
    ]
    #pg.test_agents(env, agents, filenames)