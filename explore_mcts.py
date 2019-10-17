"""
Explore the structure of an mcts.p pickle file.
"""
from datetime import date

from mcts.mcts import MCTS, joint_actions
import game
from game import envs
from game.envs import all_actions

env =  game.envs.Environment()


mcts = MCTS(env, max_search_time=10.0)

filename = 'mcts_temp.p'
mcts.load(filename)

for state in mcts.children:
    print(len(mcts.children[state]))