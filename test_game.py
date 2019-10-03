import game
from game import agents
from game.gui import visualize

from game.envs import unflatten, State

agent_list = [
    agents.RandomTank(0), # Team 1
    agents.RandomTank(1), # Team 1
    agents.RandomTank(2), # Team 2
    agents.RandomTank(3)  # Team 2
]

env = game.make(0, agent_list)
obs = env.set_init_game_state()

for i in range(0):
    actions = env.act(obs)

state = env.get_state()
flat_board = state.board
board = unflatten(flat_board, env.agents)
env.render(board)

new_state = State(
    board = state.board,
    positions = state.positions,
    alive = state.alive,
    ammo = (10, 101, 0, 5),
    aim = state.aim,
)
env.set_state(new_state)
print([agent.ammo for agent in env.agents])
