from mcts.mcts import CommandedTank, MCTSPlayer
import game

## Dilemma: agent needs player, env needs agents, player needs reference to env

player1 = MCTSPlayer()
player2 = MCTSPlayer()
agent_list = [
    CommandedTank(0, commander=player1),
    CommandedTank(1, commander=player1),
    CommandedTank(2, commander=player2),
    CommandedTank(3, commander=player2),
]

env = game.make(0, agent_list)

# add this environment to both players
for player in [player1, player2]:
    player.set_environment(env)

print(len(player1.action_space))

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
    env.render()
