import game
from game import agents
from game.envs import Environment
from marl import pg, agent_models
from game.envs import unflatten, State
import torch

import argparse

BOARD_SIZE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = pg.PGAgent(0, device, board_size=BOARD_SIZE)

agent_list = [
    agent, # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = Environment(agent_list, size=BOARD_SIZE)

if __name__ == '__main__':
    agent.set_model((1, env.board_size, env.board_size), 8,
                    lr=0.01, device=device)

    state = env.get_init_game_state()
    
    for _ in range(30):
        print(state)
        actions = [agent.get_action(state) for agent in agent_list]
        state = env.step(state, actions)
        print(actions)
        env.render(state)


