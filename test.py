import game
from game import agents

agent_list = [
    agents.Tank(0), # Team 1
    agents.Tank(1), # Team 1
    agents.Tank(2), # Team 2
    agents.Tank(3)  # Team 2
]

env =  game.make(0, agent_list)
obs = env.set_init_game_state()

env.render()

actions = (6, 0, 0, 0)
obs = env.step(actions)

env.render()