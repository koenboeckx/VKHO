from . import agent_models
from .common import *
from tensorboardX import SummaryWriter

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn
from torch.nn import functional as F

from collections import namedtuple
Experience = namedtuple('Experience', [
    'state', 'actions', 'reward', 'next_state', 'done'
])

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
        self.model = agent_models.PGModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(self.device)
    
    def get_action(self, state):
        state_v = preprocess([state]).to(self.device)
        _, logprobs = self.model(state_v)
        probs = nn.Softmax(dim=1)(logprobs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()

def generate_episode(env):
    """Generates one episode."""
    episode = []
    state = env.get_init_game_state()
    while not env.terminal(state):
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) else False
        episode.append(Experience(state, actions, reward,
                                next_state, done))
        state = next_state
    return episode

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
        cum_reward = [gamma * c + r for c, r in zip(cum_reward, reward)] # TODO: only agent 0
        returns.append(cum_reward)
    return list(reversed(returns))

#'state', 'actions', 'reward', 'next_state', 'done'
def reinforce(env, agents, **kwargs):
    """
    Apply REINFORCE to the agents. Uses environement env.
    :param env: Environment
    :param agents: list of agents to train (subset of env.agents)

    :return: None
    """
    gamma = kwargs.get('gamma', 0.99)
    n_episodes = kwargs.get('n_episodes', 100)
    agent = agents[0]

    while True:
        episodes = []
        returns  = []
        for _ in range(n_episodes):
            episode = generate_episode(env)
            returns_ = compute_returns(env, episode, gamma)
            episodes.extend(episode)
            returns.extend(returns_)
        
        n_states = len(episodes) # total number of states seen during all episodes
        episodes = list(zip(*episodes)) # reorganise episode
        returns = list(zip(*returns)) # reorganise returns

        for agent in agents:
            agent.model.optim.zero_grad()
            
            returns_v = torch.tensor(returns[agent.idx]).to(agent.device)
            states_v  = preprocess(episodes[0]).to(agent.device)
            actions_v = torch.LongTensor([a[agent.idx] for a in episodes[1]]).to(agent.device)

            _, logits_v = agent.model(states_v)
            logprob_v = F.log_softmax(logits_v, dim=1)
            #logprob_a = logprob.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            logprob_act_vals_v = returns_v * logprob_v[range(n_states), actions_v]
            loss = - logprob_act_vals_v.mean()

            print('Agent {}: Loss = {:.3f}'.format(str(agent), loss.item()))

            loss.backward()
            agent.model.optim.step()
