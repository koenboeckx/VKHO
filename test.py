import game
from game import agents

agent_list = [
    agents.TestAgent(1),
    agents.TestAgent(2), # Team 1
    agents.TestAgent(3),
    agents.TestAgent(4)  # Team 2
]

env =  game.make(0, agent_list)
obs = env.set_init_game_state()

print('ok')
env.render()