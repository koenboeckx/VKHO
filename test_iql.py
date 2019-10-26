import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model
from game.envs import unflatten, State

agent = iql.IQLAgent(0)

agent_list = [
    agent, # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list)

iql.train(env, agent, n_steps=1e3, 
            mini_batch_size=5,
            buffer_size = 20)

