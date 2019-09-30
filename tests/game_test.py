#TODO: imports don't work in terminal 

import game
from game import agents

agent_list = [
    agents.TestAgent(),
    agents.TestAgent()
]

env =  game.make(0, agent_list)
env.set_init_game_state()

env.render()