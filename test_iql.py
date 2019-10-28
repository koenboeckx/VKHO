import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model
from game.envs import unflatten, State, print_obs

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
mini_batch_size = 128
iql.train(env, [agent0, agent1],
                mini_batch_size = mini_batch_size,
                buffer_size = 4*mini_batch_size,
                sync_rate = 16*mini_batch_size,
                print_rate=1000,
                n_steps=1e10)

# TODO: train an smaller board



