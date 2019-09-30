import game
from game import agents

agent_list = [
    agents.BaseAgent(),
    agents.BaseAgent()
]

env =  game.make(0, agent_list)

env.render()