from game import envs
from game.envs import print_state, all_actions
from mcts.mcts import MCTS, MCTSStore, joint_actions

def  play_game(env, max_search_time=2.0, filename=None):
    """Play a single game"""
    mcts_stores = (MCTSStore(), MCTSStore())
    mcts = MCTS(mcts_stores, env, max_search_time=max_search_time)

    state = env.get_init_game_state()
    result = env.terminal(state)
    while result == 0: # nobody has won
        current_player = state.player
        action_idx = mcts.get_action(state)
        action = joint_actions[action_idx]
        print('Player {} plays ({}, {}) - # visited nodes = {}'.format(
            current_player, all_actions[action[0]],
            all_actions[action[1]], len(mcts_stores[current_player].n_visits)))
    
        print('UCT for state = ', mcts.uct(state))
        print('State visites = ', mcts.stores[current_player].n_visits[state])

        state = mcts.get_next(state, action)
        env.render(state)
        print_state(state)

        mcts.reset() # clear stores

    result = env.terminal(state)

env =  envs.Environment(size=5, max_range=3)
play_game(env, max_search_time=10.0)