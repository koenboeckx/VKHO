import game
from game import agents
from game.gui import visualize
from marl import iql

from game.envs import unflatten, State

agent_list = [
    iql.IQLAgent(0), # Team 1
    iql.IQLAgent(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list)
obs = env.set_init_game_state()

for i in range(10):
    actions = env.act(obs)
    print(actions)
    env.step(actions)
    env.render()