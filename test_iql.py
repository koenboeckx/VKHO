import game
from game import agents
from game.gui import visualize
from marl import iql, iql_model
from game.envs import unflatten, State, print_obs
import torch

import argparse

BOARD_SIZE = 5

agent0 = iql.IQLAgent(0, board_size=BOARD_SIZE)
agent1 = iql.IQLAgent(1, board_size=BOARD_SIZE)
agent2 = iql.IQLAgent(2, board_size=BOARD_SIZE)

agent_list = [
    agent0, # Team 1
    agents.RandomTank(1), # agent1, # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list, 
                board_size=BOARD_SIZE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", dest="train", default=True, 
                        help="Don't train model", action="store_false")
    parser.add_argument("--test", dest="test", default=False,
                        help="Test the model", action="store_true")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Enable cuda")
    parser.add_argument("--batchsize", default=256, help="Minibatch size")
    parser.add_argument("--buffersize", default=1024, help="Buffer size")
    parser.add_argument("--syncrate", default=256, help="Synchronisation rate")
    parser.add_argument("--lr", default=0.001, help="Learning rate")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    args.train = False
    if args.train:
        mini_batch_size = args.batchsize
        iql.train(env, [agent0],
                        mini_batch_size = int(args.batchsize),
                        buffer_size = int(args.buffersize),
                        sync_rate = int(args.syncrate),
                        print_rate = 500,
                        n_steps=20000,
                        lr = float(args.lr),
                        save=True)
    args.test = True
    if args.test:
        filenames = ['./marl/models/iql_agent_0_1699.torch']
                    #'./marl/models/iql_agent_1_1779.torch']
        iql.test(env, [agent0], filenames)
        print('ok')


