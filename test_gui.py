import game
from game import agents
from game.gui import visualize

agent_list = [
    agents.RandomTank(0), # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env =  game.make(0, agent_list)
visualize(env)