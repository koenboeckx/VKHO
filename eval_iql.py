"""
Evaluate and test a trained model.
"""

import game
from game import agents, envs
from game.envs import Environment
from marl import iql, iql_model
from marl.iql import preprocess
from game.envs import unflatten, State

import torch

BOARD_SIZE = 5

filename = './marl/models/iql_agent_0_70000_1248.torch'

agent = iql.IQLAgent(0, board_size=BOARD_SIZE)

agent_list = [
    agent, # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = Environment(agent_list, size=BOARD_SIZE)

input_shape = (1, BOARD_SIZE, BOARD_SIZE)
device = "cpu"
agent.set_model(input_shape, env.n_actions, 0.02, device)
agent.load_model(filename)
agent.model.eval()

state = env.get_init_game_state()
print(state)

actions = (4, 0, 0, 0)
#state = env.step(state, actions)
print(state)
env.render(state)

for step_idx in range(20):
    print('\n--------- {} -------------'.format(step_idx))
    action = torch.argmax(agent.model(preprocess([state]))).item()
    print(agent.model(preprocess([state])))
    print('action = {}: {}'.format(action, envs.all_actions[action]))
    actions = (action, 0, 0, 0)
    state = env.step(state, actions)
    env.render(state)
    print(state)
