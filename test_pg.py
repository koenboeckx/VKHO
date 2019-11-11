import game
from game import agents
from game.envs import Environment
from marl import pg, agent_models
from game.envs import unflatten, State
import torch

import argparse

BOARD_SIZE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent0 = pg.PGAgent(0, device, board_size=BOARD_SIZE)
agent1 = pg.PGAgent(1, device, board_size=BOARD_SIZE)

agent_list = [
    agent0, # Team 1
    agent1, # Team 1
    #agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]
agents = [agent0, agent1]
env = Environment(agent_list, size=BOARD_SIZE)

if __name__ == '__main__':
    for agent in agents:
        agent.set_model((1, env.board_size, env.board_size), env.n_actions,
                        lr=0.001)
    
    pg.reinforce(env, agents)