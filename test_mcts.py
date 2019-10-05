"""
Implementation with only one player, who in turn steers team1 and then
team 2. Turns game with simultaneous actions into turn-based game.
"""

from mcts.mcts import CommandedTank, MCTSPlayer
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env

player = MCTSPlayer()
agent_list = [
    CommandedTank(0, commander=player),
    CommandedTank(1, commander=player),
    CommandedTank(2, commander=player),
    CommandedTank(3, commander=player),
]

env = game.make(0, agent_list)

# add this environment to player
player.set_environment(env)
player.init_stores()

print(len(player.action_space))

obs = env.set_init_game_state()

#actions = env.act(obs) # env.act returns tuple of actions
                        # this doesn't have to be used.
                        # use players to generate actions

state = env.get_state()
player.get_actions(0, state)

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