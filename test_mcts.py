"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""
from datetime import date

from mcts.mcts import MCTS, joint_actions
import game
from game import envs
from game.envs import all_actions

env =  game.envs.Environment(size=5, max_range=3)

mcts = MCTS(env, max_search_time=2.0)

filename = 'mcts_{}.p'.format(date.today())
#filename = 'mcts_temp.p'
try:
    mcts.load(filename)
except FileNotFoundError:
    pass

state = env.get_init_game_state()
for it in range(200):
    print('iteration {}'.format(it))
    action_idx = mcts.get_action(state)
    action = joint_actions[action_idx]
    print('Player {} plays ({}, {}) - # visited nodes = {}'.format(
        state.player, all_actions[action[0]],
        all_actions[action[1]], len(mcts.n_visits)))
    print('UCT for state = ', sorted(mcts.uct(state), reverse=True))

    state = mcts.get_next(state, action)
    env.render(state)
    game.envs.print_state(state)

    if env.terminal(state):
        print('Game won by player {}'.format(state.player))
        state = env.get_init_game_state()
mcts.save(filename)

print('... done')
