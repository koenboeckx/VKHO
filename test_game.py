import random

import game
from game import agents
from game.gui import visualize

from game.envs import unflatten, State, Environment

agent_list = [
    agents.RandomTank(0), # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = Environment(agent_list)
state = env.get_init_game_state()

env.render(state)

actions = (7, 0, 0, 0)
state = env.step(state, actions)
actions = (7, 0, 0, 0)
state = env.step(state, actions)
actions = (5, 0, 0, 0)
state = env.step(state, actions)
actions = (5, 0, 0, 0)
state = env.step(state, actions)

env.render(state)

actions = (0, 2, 0, 0)
state = env.step(state, actions)

actions = (0, 3, 0, 0)
state = env.step(state, actions)

actions = (4, 3, 0, 0)
state = env.step(state, actions)

actions = (4, 3, 0, 0)
state = env.step(state, actions)

env.render(state)
print(state)