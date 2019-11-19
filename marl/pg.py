from . import agent_models
#from .common import preprocess_extended as preprocess
from .common import preprocess_gym as preprocess

from tensorboardX import SummaryWriter
from collections import namedtuple
from datetime import datetime

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.nn.utils as nn_utils

from tensorboardX import SummaryWriter
import numpy as np

SAVE_RATE = 1000
REWARDS_STEPS = 1 # don't just use next step in bootstrapping, but look ahead NOT USED YET
CLIP_GRAD = 0.1

Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])
# Experience.__new__.__defaults__ = (None, ) * len(Experience._fields) # set default values


class PGAgent(agents.Tank):
    """
    Agent class to be used with policy gradient RL.
    Model is set with method .set_model()
    """
    def __init__(self, idx, device, board_size=11):
        super(PGAgent, self).__init__(idx)
        self.init_agent(idx)
        self.board_size = board_size
        self.device = device
    
    def set_model(self, input_shape, n_actions, lr):
        """
        Set the model (neural net) of the agent.

        :param input_shape: shape of input vector (channels, widht, height)
        :param lr:          learning rate for model optimizer (=Adam)
        :param device:      torch.device("cpu" or "cuda")
        :return: None
        """
        self.model = agent_models.PGExtendedModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(self.device)
    
    def save_model(self, filename=None):
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = '/home/koen/Programming/VKHO/marl/models/pg_agent{}_{}.torch'.format(
                            self.idx, date_str)
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def get_action(self, state):
        board_v, state_v = [tensor.to(self.device) for tensor in preprocess([state])]
        _, logits = self.model(board_v, state_v)
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

class GymAgent:
    def __init__(self, idx, device, **kwargs):
        self.idx = idx
        self.device = device
    
    def __repr__(self):
        return 'gym{}'.format(self.idx)

    def set_model(self, input_shape, n_actions, n_hidden, lr):
        self.model = agent_models.GymModel(input_shape, n_actions,
            n_hidden=n_hidden, lr=lr).to(self.device)
    
    def save_model(self, filename=None):
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = '/home/koen/Programming/VKHO/marl/models/gym_pg_agent{}_{}.torch'.format(
                            self.idx, date_str)
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def get_action(self, state):
        state_v = [t.to(self.device) for t in preprocess([state])]
        _, logits = self.model(*state_v)
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

def generate_episode(env):
    """Generates one episode."""
    episode = []
    state = env.get_init_game_state()
    while env.terminal(state) == 0:
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) else False
        episode.append(Experience(state, actions, reward,
                                next_state, done))
        state = next_state
    return episode

def generate_steps(env, n_steps, gamma=0.99):
    """
    Generate n_steps through interaction with environment env. Actions are
    picked with agent.get_action(state) methods for the agents in env.agents.
    """
    experiences = []
    returns = []
    state = env.get_init_game_state()
    episode = []
    for _ in range(n_steps):
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) else False
        episode.append(Experience(state, actions, reward, next_state, done))
        if done: # end of episode
            experiences.extend(episode)
            returns.extend(compute_returns(env, episode, gamma)) # only compute returns at end of episode
            episode = []
            state = env.get_init_game_state() # reinit game when episode is done
        else:
            state = next_state
        
    return experiences, returns
            
def compute_returns(env, episode, gamma):
    """Compute discounted cumulative rewards for all agents

    :param env:     Environment containing the agents
    :param epsiode: list of Experiences with last one the final of the episode
    :params gamma:  discount factor

    :return: list of discounted cumulative rewards [..., (G0, G1, G2, G3), ...]
    """
    returns = []
    cum_reward = [0.0,] * len(env.agents)
    for _, _, reward, _, _ in reversed(episode):
        cum_reward = [gamma * c + r for c, r in zip(cum_reward, reward)]
        returns.append(cum_reward)
    return list(reversed(returns))

def compute_mean_reward(episodes, agent):
    "Compute mean reward for agent"
    rewards = 0.0
    n_episodes = 0
    for _, _, reward, _, done in episodes:
        rewards += reward[agent.idx]
        if done:
            n_episodes += 1
    return len(episodes)/n_episodes, rewards/n_episodes

def reinforce(env, agents, **kwargs):
    """
    Apply REINFORCE to the agents. Uses environement env.
    :param env: Environment
    :param agents: list of agents to train (subset of env.agents)

    :return: None
    """
    gamma = kwargs.get('gamma', 0.99)
    n_steps = kwargs.get('n_steps', int(1e5))
    n_episodes = kwargs.get('n_episodes', 100)
    ex = kwargs.get('experiment')

    for step_idx in range(n_steps):
        episodes = []
        returns  = []
        for _ in range(n_episodes):
            episode = generate_episode(env)
            returns_ = compute_returns(env, episode, gamma)
            episodes.extend(episode)
            returns.extend(returns_)
            
        mean_length, mean_reward = compute_mean_reward(episodes, agents[0])
        ex.log_scalar("mean_reward", mean_reward)
        ex.log_scalar("mean_length", mean_length)

        n_states = len(episodes) # total number of states seen during all episodes
        episodes = list(zip(*episodes)) # reorganise episode
        returns = list(zip(*returns)) # reorganise returns

        for agent in agents:
            agent.model.optimizer.zero_grad()
            
            returns_v = torch.tensor(returns[agent.idx]).to(agent.device)
            states_v  = [tensor.to(agent.device) for tensor in preprocess(episodes[0])]
            actions_v = torch.LongTensor([a[agent.idx] for a in episodes[1]]).to(agent.device)

            _, logits_v = agent.model(*states_v)
            logprob_v = F.log_softmax(logits_v, dim=1)
            logprob_act_vals_v = returns_v * logprob_v[range(n_states), actions_v]
            loss = - logprob_act_vals_v.mean()

            print('Agent {}: Loss = {:.5f}'.format(str(agent), loss.item()))
            ex.log_scalar('loss_{}'.format(str(agent)), loss.item())

            loss.backward()
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in agent.model.parameters()
                                    if p.grad is not None]
            )
            ex.log_scalar('grad_l2_{}'.format(agent),  np.sqrt(np.mean(np.square(grads))))

            agent.model.optimizer.step()

        if step_idx > 0 and step_idx % SAVE_RATE == 0:
            agent.save_model()

def actor_critic_(env, agents, **kwargs):
    """
    Use Actor-Critic learning to train agents. Uses environement env.
    :param env: Environment
    :param agents: list of agents to train (subset of env.agents)

    :return: None
    """
    gamma = kwargs.get('gamma', 0.99)           # discount factor
    n_steps = kwargs.get('n_steps', 100)        # number of steps to generate each iteration
    batch_size = kwargs.get('batch_size', 32)   # number of exp used for learning
    ex = kwargs.get('experiment')

    idx = 0
    for step_idx in range(n_steps):
        exp_source, returns = generate_steps(env, 8*batch_size, gamma) # TODO: IDEA: make this a generator (-> yield)
        batch = []

        mean_length, mean_reward = compute_mean_reward(exp_source, agents[0])
        ex.log_scalar('mean_reward', mean_reward)
        ex.log_scalar('mean_length', mean_length)

        print('Step {:3d} - mean length = {:.2f}, mean reward = {:.3f}'.format(
            step_idx, mean_length, mean_reward
        ))

        for exp, ret  in zip(exp_source, returns):
            # batch.append((exp, ret))
            batch.append((exp, exp.reward))
            if len(batch) < batch_size:
                continue

            for agent in agents:
                states_v, actions_t, vals_ref_v = unpack_batch(env, agent, batch, gamma)
                    
                agent.model.optimizer.zero_grad()

                values_v, logits_v = agent.model(*states_v)
                loss_value_v = F.mse_loss(values_v.squeeze(-1), vals_ref_v) # loss for value estimate

                logprob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - values_v.detach().squeeze(-1)
                logprob_actions_v = adv_v * logprob_v[range(batch_size), actions_t]
                loss_policy_v = -logprob_actions_v.mean()   # policy gradient

                # TODO: add entropy loss
                    
                loss_policy_v.backward(retain_graph=True)

                # store grads for analysis
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten() 
                                        for p in agent.model.parameters()
                                        if p.grad is not None])
                
                ex.log_scalar('grad_l2_{}'.format(agent),
                                    np.sqrt(np.mean(np.square(grads))))
                ex.log_scalar('loss_value_{}'.format(agent), loss_value_v.item())

                loss_value_v.backward()
                    
                # clip gradients
                nn.utils.clip_grad_norm_(agent.model.parameters(),
                                        CLIP_GRAD)
                agent.model.optimizer.step()
            
            idx += 1
            batch.clear()

def unpack_batch(env, agent, batch, gamma):
    """
    Convert batch into training tensors
    :param env:
    :param agent:
    :param batch: list: [..., (exp, return), ...]
    :param gamma:
    :return: (boards variable, states variable), actions tensor, reference values variable
    """
    states, actions, rewards, not_done_idx, last_states = [], [], [], [], []
    for idx, (exp, ret) in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.actions[agent.idx])
        rewards.append(exp.reward[agent.idx]) 
        
        # store indexes where episode is not done
        if not exp.done: # TODO: change Experience class to allow longer rollouts
            not_done_idx.append(idx)
            last_states.append(exp.next_state)

    states_v = [t.to(agent.device) for t  in preprocess(states)]
    actions_t = torch.LongTensor(actions).to(agent.device)

    # handle the rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx: # check if not_done is present and then update these states with V(s_t+1)
        last_states_v  = [t.to(agent.device) for t in preprocess(last_states)]
        last_vals_v, _ = agent.model(*last_states_v)
        last_vals_np   = last_vals_v.data.cpu().numpy().squeeze()
        rewards_np[not_done_idx] += gamma * last_vals_np
    ref_vals_v = torch.tensor(rewards_np).to(agent.device)

    return states_v, actions_t, ref_vals_v

def test_agents(env, agents, filenames):
    for agent, filename in zip(agents, filenames):
        agent.load_model(filename)
    
    state = env.get_init_game_state()
    while not env.terminal(state):
        actions = [agent.get_action(state) for agent in env.agents]
        print(actions)
        next_state = env.step(state, actions)
        env.render(next_state)
        print(next_state)

        state = next_state

# new try to implement actor-critic
def actor_critic(env, agents, **kwargs):
    """
    Use Actor-Critic learning to train agents. Uses environement env.
    :param env: Environment
    :param agents: list of agents to train (subset of env.agents)

    :return: None
    """

    gamma = kwargs.get('gamma', 0.99)           # discount factor
    n_steps = kwargs.get('n_steps', 100)
    n_episodes = kwargs.get('n_episodes', 100)
    ex = kwargs.get('experiment')

    # 1. init network
    for step_idx in range(n_steps):
        # 2. Play N steps in env using current policy -> store experience (s_t, a_t, r_t, s_t+1)
        episodes = []
        for _ in range(n_episodes):
            state = env.get_init_game_state()
            while not env.terminal(state):
                #env.render(state)
                actions = [agent.get_action(state) for agent in env.agents]
                next_state = env.step(state, actions)
                reward = env.get_reward(next_state)
                done = True if env.terminal(next_state) else False
                episodes.append(Experience(state, actions, reward, next_state, done))
                
                state = next_state

        # discount states
        end_states = [-1] + [idx for idx, exp in enumerate(episodes) if exp.done]
        returns = [0.0,] * len(episodes)
        for start, stop in zip(end_states[:-1], end_states[1:]):
            cum_reward = [0.0, ] * len(env.agents)
            for j in range(stop, start, -1):
                exp = episodes[j]
                cum_reward = [gamma * c + r for c, r in zip(cum_reward, exp.reward)]
                returns[j] = cum_reward
            
        
        episodes = list(zip(*episodes)) # reorganise episodes
        states, actions, rewards, next_states, dones = episodes

        mean_length = len(episodes[0]) / n_episodes
        mean_reward = sum([reward[0] for reward in rewards]) / n_episodes
        print('Episode length = {}'.format(mean_length))
        print('Episode reward = {}'.format(mean_reward))

        ex.log_scalar('mean_length', mean_length)
        ex.log_scalar('mean_reward', mean_reward)

        for agent in agents:
            states_v  = [t.to(agent.device) for t in preprocess(states)]
            next_states_v = [t.to(agent.device) for t in preprocess(next_states)]
            actions_t = torch.LongTensor([action[agent.idx]
                                            for action in actions]).to(agent.device)
            dones_t = torch.LongTensor(dones)

            values_v, logits_v = agent.model(*states_v)
            vals_ref_v, _ = agent.model(*next_states_v)

            vals_ref_v = vals_ref_v.squeeze(-1)
            vals_ref_v[dones_t == 1.0] = 0.0 # set done states to zero

            #rewards_v = torch.tensor([reward[agent.idx] for reward in rewards])
            rewards_v = torch.tensor([ret[agent.idx] for ret in returns])
            vals_ref_v = rewards_v + gamma * vals_ref_v

            loss_values_v = F.mse_loss(values_v.squeeze(-1), vals_ref_v)

            logprobs_v = F.log_softmax(logits_v, dim=1)
            advantage_v = vals_ref_v - values_v.squeeze(-1).detach()
            log_prob_actions_v = advantage_v * logprobs_v[range(len(states)), actions_t]
            loss_policy_v = -log_prob_actions_v.mean()
            
            agent.model.zero_grad()
            loss_policy_v.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in agent.model.parameters()
                                    if p.grad is not None]
            )

            loss_values_v.backward()

            nn_utils.clip_grad_norm_(agent.model.parameters(), CLIP_GRAD) # clip gradients
            agent.model.optimizer.step()

            ## bookkeeping
            ex.log_scalar('loss_value_{}'.format(agent), loss_values_v.item())
            ex.log_scalar('grad_l2_{}'.format(agent),  np.sqrt(np.mean(np.square(grads))))
