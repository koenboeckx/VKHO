"""
Learn a strategy with independent Q-learning
"""

import game
import time
from tensorboardX import SummaryWriter

TEMP = 0.4
ALPHA = 0.8
GAMMA = 0.9

N_HUNTERS, N_PREY = 2, 2

def train(env, hunters, prey, n_episodes=100):
    with SummaryWriter(comment='_tan') as writer:
        for episode in range(n_episodes):
            state, obs = env.get_init_state()
            while not env.terminal():
                h_actions = [h.get_action(obs[h], TEMP) for h in hunters]
                p_actions = [p.get_action(0) for p in prey]
                next_state, next_obs = env.step(state, h_actions, p_actions)
                reward = env.get_reward()

                # update Q-values
                for idx, hunter in enumerate(hunters):
                    predicted_v = hunter.Q[obs[hunter]][h_actions[idx]]
                    if reward == 1: # episode is terminated:
                        target_v = reward
                    else:
                        target_v = reward + GAMMA * max(hunter.Q[next_obs[hunter]])
                    hunter.Q[obs[hunter]][h_actions[idx]] += ALPHA * (target_v - predicted_v)
                
                obs = next_obs
                state = next_state
            
            n_eval = eval(env, hunters, prey)
            #print('After episode {:4} -> {:3} steps required'.format(episode, n_eval))
            #print(hunters[0].Q[(1, 0)])
            #time.sleep(1.)
            writer.add_scalar('steps', n_eval, episode)

def eval(env, hunters, prey):
    n_steps = 0

    state, obs = env.get_init_state()
    while not env.terminal():
        h_actions = [h.get_action(obs[h], TEMP) for h in hunters]
        p_actions = [p.get_action(0) for p in prey]
        next_state, next_obs = env.step(state, h_actions, p_actions)
        n_steps += 1
        obs = next_obs
        state = next_state
    
    return n_steps



if __name__ == '__main__':
    hunters, prey, env = game.create_game(N_HUNTERS, N_PREY)
    train(env, hunters, prey, n_episodes=10000)
    print('ok')