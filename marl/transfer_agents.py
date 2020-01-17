"""
Experiment with transfering agent knowledge
"""
import sys; sys.path.insert(1, '/home/koen/Programming/VKHO/game')

import pickle

from envs import Environment, all_actions
from gui import visualize
from pg2 import A2CAgent, A2CModel, RandomTank, params

def visualize_game(env,agents):
    state = env.get_init_game_state()
    while not env.terminal(state):
        env.render(state)
        actions = [agent.get_action(state) for agent in agents]
        print(f"Actions = {[all_actions[a] for a in actions]}")
        state = env.step(state, actions)


def run():
    learners = []
    for agent_idx in [0, 1]:
        with open(f'agent{agent_idx}.pkl', 'rb') as input_file:
            learners.append(pickle.load(input_file))
    opponents = [RandomTank(idx) for idx in [2, 3]]
    agents = learners + opponents
    env = Environment(agents, size=params['board_size'])
    #visualize_game(env, agents)
    visualize(env, period=1)

if __name__ == '__main__':
    run()