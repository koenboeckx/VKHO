"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import MCTS, joint_actions
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env

from game import envs

env =  game.envs.Environment()
state = env.get_init_game_state()

mcts = MCTS(env, max_search_time=1.0)

for i in range(10):
    action_idx = mcts.get_action(state)
    action = joint_actions[action_idx]
    print('Player {} plays {} - # visited nodes = {}'.format(
        state.player, action, len(mcts.n_visits)))
    if state.player == 0:
        actions = action + (0, 0)
    else:
        actions = (0, 0) + action
    state = env.step(state, actions)
    env.render(state)
    game.envs.print_state(state)

print('... done')
