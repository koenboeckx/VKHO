"""
Experiment with transfering agent knowledge
"""
import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

import pickle

import torch

from envs import Environment, all_actions
from gui import visualize, create_gif
from pg2 import A2CAgent, A2CModel, RandomTank, params, train

STORE = False
if STORE:
    from sacred import Experiment
    from sacred.observers import MongoObserver
    ex = Experiment('new_pg')
    ex.observers.append(MongoObserver(url='localhost',
                                    db_name='my_database'))

def visualize_game(env,agents):
    state = env.get_init_game_state()
    step_idx = 0
    while not env.terminal(state):
        env.render(state)
        actions = env.get_actions(state)
        print(f"Actions = {[all_actions[a] for a in actions]}")
        state = env.step(state, actions)
        step_idx += 1
    print(state)
    print(f'Game won by team {0 if env.terminal(state) == 1 else 1} in {step_idx} steps.')


def run():
    learners = []
    for agent_idx in [0, 1]:
        with open(f'agent{agent_idx}-with_penalty.pkl', 'rb') as input_file:
            learners.append(pickle.load(input_file))
    opponents = [RandomTank(idx) for idx in [2, 3]]
    agents = learners + opponents
    env = Environment(agents, size=params['board_size'])
    #visualize_game(env, agents)
    visualize(env, period=0.1)

if __name__ == '__main__':
    run()
    create_gif('./game/screenshots/')
