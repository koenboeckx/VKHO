import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model
from game.envs import unflatten, State

agent0 = iql.IQLAgent(0)
agent1 = iql.IQLAgent(1)
agent2 = iql.IQLAgent(2)

agent_list = [
    agent0, # Team 1
    agent1, # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list)

iql.train(env, [agent0, agent1], print_rate=100,
         n_steps=1e10)

# TODO: train an samller board



