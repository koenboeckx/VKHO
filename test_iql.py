import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model
from game.envs import unflatten, State, print_obs

BOARD_SIZE = 5
TRAIN = True # set true if you want to train
TEST  = False  # set true if you want to test

agent0 = iql.IQLAgent(0, board_size=BOARD_SIZE)
agent1 = iql.IQLAgent(1, board_size=BOARD_SIZE)
agent2 = iql.IQLAgent(2, board_size=BOARD_SIZE)

agent_list = [
    agent0, # Team 1
    agent1, # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list, 
                board_size=BOARD_SIZE)

if TRAIN:
    mini_batch_size = 32
    iql.train(env, [agent0, agent1],
                    mini_batch_size = mini_batch_size,
                    buffer_size = 4*mini_batch_size,
                    sync_rate = 16*mini_batch_size,
                    print_rate = 500,
                    n_steps=3000,
                    save=True)
if TEST:
    filenames = ['./marl/models/iql_agent_0_1322.torch',
                 './marl/models/iql_agent_1_1322.torch']
    iql.test(env, [agent0, agent1], filenames)
    print('ok')


