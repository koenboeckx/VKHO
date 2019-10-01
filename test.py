import game
from game import agents

agent_list = [
    agents.RandomTank(0), # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env =  game.make(0, agent_list)
obs = env.set_init_game_state()

env.render()

for i in range(2):
    actions = env.act(obs)
    obs = env.step(actions)
    env.render()