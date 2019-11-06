"""
Learn a strategy with independent Q-learning
"""

import game
import time
from tensorboardX import SummaryWriter
import argparse

TEMP = 0.4
ALPHA = 0.2
GAMMA = 0.9
SYNC_RATE = 50

N_HUNTERS, N_PREY = 2, 1
DEPTH = 2

VISUALIZE = False # set to True to show action distribution of all states of hunter0

def temp_schedule(start, stop, n_steps):
    """Returns a (linear) temperature schedule."""
    def temperature(step):
        if step > n_steps:
            return stop
        else:
            return start + (stop - start) / n_steps * step
    return temperature

def train(env, hunters, prey, n_steps=100, verbose=False):
    writer_comment = '_tan_agents_{}_depth_{}'.format(len(hunters), hunters[0].depth)
    with SummaryWriter(comment=writer_comment) as writer:
        total_steps = 0
        temperature = temp_schedule(0.4, 0.01, 10000)
        for step in range(n_steps):
            temp = temperature(step)
            temp = 0.4 # TODO: comment this line
            state, observations = env.get_init_state()
            
            h_actions = [h.get_action(observations[h], temp) for h in hunters]
            p_actions = [p.get_action(None) for p in prey] # here, obs doesn't matter
            
            next_state, next_observations = env.step(state, h_actions, p_actions)
            reward = env.get_reward(next_state)

            # update Q-values
            for hunter, action,  in zip(hunters, h_actions):
                obs, next_obs = observations[hunter], next_observations[hunter]
                predicted_v = hunter.Q[obs][action]
                if env.terminal(next_state): # episode is terminated:
                    target_v = reward
                else:
                    target_v = reward + GAMMA * max(hunter.Q[next_obs])
                hunter.Q[obs][action] += ALPHA * (target_v - predicted_v)
                
            if env.terminal(next_state):
                next_state, next_observations = env.get_init_state()
                
            observations = next_observations
            state = next_state
            
            # bookkeeping and performance eval
            n_eval = eval(env, hunters, prey)           
            total_steps += n_eval

            if step > 0 and step % SYNC_RATE == 0:
                if verbose is True:
                    print('Step {:4d} -> avg. {:8.1f} steps required'.format(step,
                                                                        total_steps/SYNC_RATE))
                writer.add_scalar('steps', total_steps/SYNC_RATE, step)
                total_steps = 0

                if VISUALIZE:
                    game.visualize(hunters[0], temp, title='# steps = {}'.format(step))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hunters", default=2, help="Number of hunters")
    parser.add_argument("--prey", default=1, help="Number of prey")
    parser.add_argument("--depth", default=2, help="Depth")
    parser.add_argument("--verbose", dest='verbose', 
                        default=False, action="store_true")
    args = parser.parse_args()

    hunters, prey, env = game.create_game(int(args.hunters),
                                          int(args.prey),
                                          int(args.depth))
    train(env, hunters, prey, n_steps=100000, verbose=args.verbose)