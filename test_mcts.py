"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import CommandedTank, MCTS, Player, joint_actions
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env

from game import envs

env =  game.envs.Environment()
state = env.get_init_game_state()

player1 = Player(0, env)
player2 = Player(1, env)

mcts = MCTS(player1, player2, max_search_time=1.0)
players = (player1, player2)

player_idx = 0
for i in range(100):
    print(i)
    player = players[player_idx]
    action_idx = mcts.get_action(player, state)
    action = joint_actions[action_idx]
    print('Player {} plays {} - # visited nodes = {}'.format(
        player_idx, action, len(player.n_visits)))
    if player_idx == 0:
        actions = action + (0, 0)
    else:
        actions = (0, 0) + action
    state = env.step(state, actions)
    player_idx = 1-player_idx
    env.render(state)
    game.envs.print_state(state)

print('... done')
