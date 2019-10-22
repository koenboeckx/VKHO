from game import envs
from mcts.mcts import play_game

env =  envs.Environment(size=5, max_range=3)
play_game(env)