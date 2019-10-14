"""
Implementation with only two players, who take turnst making moves.
Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import CommandedTank, MCTS, Player
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env


agent_list = [
    CommandedTank(0),
    CommandedTank(1),
    CommandedTank(2),
    CommandedTank(3),
]

env = game.make(0, agent_list)
state = env.get_state()

player1 = Player(0, env)
player2 = Player(1, env)


mcts = MCTS(player1, player2)
mcts.one_iteration(player1, state)

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