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

for i in range(10):
    obs1, obs2 = obs[:2], obs[2:] # split the global observation per team
    actions1 = player1.get_actions(obs1)
    actions2 = player2.get_actions(obs2)
    actions = actions1 + actions2
    obs = env.step(actions)
    result = env.terminal() # check if game is over. If yes => reward palyers
    if result == 1:
        pass
    env.render()