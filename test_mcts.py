"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import CommandedTank, MCTS, Player, joint_actions
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env


agent_list = [
    CommandedTank(0),
    CommandedTank(1),
    CommandedTank(2),
    CommandedTank(3),
]

env = game.make(0, agent_list)
_ = env.set_init_game_state()
state = env.get_state()

player1 = Player(0, env)
player2 = Player(1, env)


mcts = MCTS(player1, player2, max_search_time=0.1)
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
    state = env.sim_step(state, actions)
    player_idx = 1-player_idx
    env.render(state.board)

print('...')

"""
# training sequence
state = env.get_state()
team = 0
for i in range(10):
    actions = player.get_actions(team, state)
    if team == 0:
        actions = actions + (0, 0)
    else:
        actions = (0, 0) + actions
    _ = env.step(actions)
    state = env.get_state()
    if env.terminal() != 0: # check if game is over => reset game
        state = env.set_init_game_state()
    env.render()

    team = 0 if team == 1 else 1 # switch teams
"""