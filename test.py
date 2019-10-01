import game
from game import agents

agent_list = [
    agents.Tank(1),
    agents.Tank(2), # Team 1
    agents.Tank(3),
    agents.Tank(4)  # Team 2
]

env =  game.make(0, agent_list)
obs = env.set_init_game_state()

env.render()

actions = (1, 0, 0, 0)
env.step(actions)

#env.render()