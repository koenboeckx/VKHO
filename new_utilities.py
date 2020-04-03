import copy
from collections import namedtuple
import torch

from env import Observation

import yaml
from utilities import get_args
args = get_args(yaml.load(open('default_config.yaml', 'r')))

Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations',
    'hidden', 'next_obs', 'next_hidden','unavailable_actions'
])

def transform_obs(observations):
    """transform observation into tensor  of size N for storage and use in model
    N = own pos (2), alive, friends alive + pos (3*Nf) , enemies alive + pos (3*Ne), ammo (1), aim (1), enemy_visibility (Ne)"""
    result = []
    for agent in observations:
        obs = observations[agent]
        if isinstance(obs, Observation):
            N = 4 + 3 * len(obs.friends) + 3 * len(obs.enemies) + len(obs.enemy_visibility) 
            x = torch.zeros((N)) 
            x[0:2] = torch.tensor(obs.own_position)
            #x[2]   = obs.alive
            #idx = 3
            idx = 2
            for friend in obs.friends:
                if friend:  # friend is alive
                    x[idx:idx+3] = torch.tensor([1.,] + list(friend[:2]))
                else:       # friend is dead
                    x[idx:idx+3] = torch.tensor([0.,] + list(friend[:2]))
                idx += 3
            for enemy in obs.enemies:
                if enemy[2]:   # enemy is alive
                    x[idx:idx+3] = torch.tensor([1.,] + list(enemy[:2]))
                else:       # enemy is dead
                    x[idx:idx+3] = torch.tensor([0.,] + list(enemy[:2]))
                idx += 3
            
            # add enemy visibility                
            for visible in obs.enemy_visibility:
                x[idx] = torch.tensor(int(visible))
                idx += 1
            
            x[idx]   = obs.ammo / 5. #args.init_ammo
            x[idx+1] = obs.aim.id if obs.aim is not None else -1
            result.append(x)
        else:
            raise ValueError(f"'obs' should be an Observation, and is a {type(obs)}")
    return torch.stack(result)

def transform_state(state):
    """Transforms State into tensor"""
    # TODO: automate n_enemies calculation -> only valid fot n_enemies = n_friends
    n_agents  = len(state.agents)
    n_enemies = n_agents // 2 # TODO: improve this
    states_v = torch.zeros(n_agents, 5 + n_enemies) # 5 = x, y, alive, ammo, aim, enemy visible ? (x n_enemies)
    for agent_idx, agent in enumerate(state.agents):
        states_v[agent_idx, 0] = state.position[agent][0] # x
        states_v[agent_idx, 1] = state.position[agent][1] # y
        states_v[agent_idx, 2] = state.alive[agent]
        states_v[agent_idx, 3] = state.ammo[agent] / 5 # args.ammo
        states_v[agent_idx, 4] = -1 if state.aim[agent] is None else state.aim[agent].id
        idx = 5
        for other in state.agents:
            if (agent, other) in state.visible:
                states_v[agent_idx, idx] = int(state.visible[(agent, other)])
                idx += 1
    return states_v

def generate_episode(env, args, render=False, test_mode=False):
    """Generate episode; store observations and states as tensors
        render: renders every step of episode
        test_mode: picks best action, not based on epsilon-greedy
    """
    episode = []
    state, done = env.reset(), False
    observations = transform_obs(env.get_all_observations())
    n_steps = 0

    for agent in env.agents:        # for agents where it matters,
        agent.set_hidden_state()    # set the init hidden state of the RNN

    while not done:
        unavailable_actions = env.get_unavailable_actions()
        
        # compute action, keep record of hidden state of the agents to store in experience
        actions, hidden, next_hidden = {}, [], []
        for idx, agent in enumerate(env.agents):
            hidden.append(agent.get_hidden_state())
            actions[agent] = agent.act(observations[idx, :], test_mode=test_mode)
            next_hidden.append(agent.get_hidden_state())

        if render:
            print(f"Step {n_steps}")
            env.render()
            print([action.name for action in actions.values()])

        next_state, rewards, done, _ = env.step(actions)
        next_obs = transform_obs(env.get_all_observations())
        
        # episodes that take long are not allowed and penalized for both agents
        n_steps += 1
        if n_steps > args.max_episode_length:
            done = True
            rewards = {'blue': -1, 'red': -1}

        actions = torch.tensor([action.id for action in actions.values()])
        unavail_actions = torch.zeros((args.n_agents, args.n_actions), dtype=torch.long)
        for idx, agent in enumerate(env.agents):
            act_ids = [act.id for act in unavailable_actions[agent]]
            unavail_actions[idx, act_ids] = 1.

        episode.append(Experience(transform_state(state), actions, rewards, 
                                  transform_state(next_state), done, 
                                  observations, torch.stack(hidden), 
                                  next_obs, torch.stack(next_hidden),
                                  unavail_actions))
        state = next_state
        observations = next_obs
    
    if render:
        print(f"Game won by team {env.terminal(next_state)}")
    return episode
