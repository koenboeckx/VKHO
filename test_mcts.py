"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import MCTS, joint_actions
import game
from game.envs import all_actions

## Dilemma: agent needs player, env needs agents, player needs reference to env

from game import envs

env =  game.envs.Environment()
state = env.get_init_game_state()

mcts = MCTS(env, max_search_time=10.0)

for i in range(1000):
    action_idx = mcts.get_action(state)
    action = joint_actions[action_idx]
    print('Player {} plays ({}, {}) - # visited nodes = {}'.format(
        state.player, all_actions[action[0]],
        all_actions[action[1]], len(mcts.n_visits)))

    state = mcts.get_next(state, action)
    env.render(state)
    game.envs.print_state(state)

print('... done')
