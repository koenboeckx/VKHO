from game import envs

env =  envs.Environment()
state = env.get_init_game_state()

env.render(state)

actions = [ (1, 1, 1, 2),
            (4, 1, 0, 0),
            (7, 1, 6, 0),
            (3, 1, 6, 0),
]

for action in actions:
    state = env.step(state, action)
    print(state)
    env.render(state)
    envs.print_state(state)