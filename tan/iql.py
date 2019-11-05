"""
Learn a strategy with independent Q-learning
"""

import game
import time
from tensorboardX import SummaryWriter

TEMP = 0.4
ALPHA = 0.2
GAMMA = 0.9
SYNC_RATE = 20

N_HUNTERS, N_PREY = 1, 1
DEPTH = 1

def train(env, hunters, prey, n_episodes=100):
    with SummaryWriter(comment='_tan') as writer:
        total_steps = 0
        for episode in range(n_episodes):
            state, obs = env.get_init_state()
            while not env.terminal(state):
                h_actions = [h.get_action(obs[h], TEMP) for h in hunters]
                p_actions = [p.get_action(None) for p in prey] # here, obs doesn't matter
                next_state, next_obs = env.step(state, h_actions, p_actions)
                reward = env.get_reward(next_state)

                # update Q-values
                for idx, hunter in enumerate(hunters):
                    predicted_v = hunter.Q[obs[hunter]][h_actions[idx]]
                    if env.terminal(next_state): # episode is terminated:
                        target_v = reward
                    else:
                        target_v = reward + GAMMA * max(hunter.Q[next_obs[hunter]])
                    hunter.Q[obs[hunter]][h_actions[idx]] += ALPHA * (target_v - predicted_v)
                
                obs = next_obs
                state = next_state
            
            n_eval = eval(env, hunters, prey)
            writer.add_scalar('steps', n_eval, episode)
            
            total_steps += n_eval

            if episode % SYNC_RATE == 0:
                print('After episode {:4} -> avg. {:3} steps required'.format(episode,
                                                                            total_steps/SYNC_RATE))
                print('Qs[(1, 0)] = ', hunters[0].Q[(1, 0)])
                print('probs    = ', game.boltzmann(hunters[0].Q[(1, 0)], TEMP))
                total_steps = 0

                #game.visualize(hunters[0])

def eval(env, hunters, prey):
    n_steps = 0

    state, obs = env.get_init_state()
    while not env.terminal(state):
        h_actions = [h.get_action(obs[h], TEMP) for h in hunters]
        p_actions = [p.get_action(0) for p in prey]
        next_state, next_obs = env.step(state, h_actions, p_actions)
        n_steps += 1
        obs = next_obs
        state = next_state
    
    return n_steps



if __name__ == '__main__':
    hunters, prey, env = game.create_game(N_HUNTERS, N_PREY, DEPTH)
    train(env, hunters, prey, n_episodes=100000)