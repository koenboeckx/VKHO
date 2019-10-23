import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model

from game.envs import unflatten, State

agent_list = [
    iql.IQLAgent(0), # Team 1
    iql.IQLAgent(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list)
iql.train(env, agent_list[0])