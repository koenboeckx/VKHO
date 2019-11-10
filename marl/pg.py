from . import agent_models
from .common import *
from tensorboardX import SummaryWriter

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn

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
    
    def set_model(self, input_shape, n_actions, lr, device):
        """
        Set the model (neural net) of the agent.

        :param input_shape: shape of input vector (channels, widht, height)
        :param lr:          learning rate for model optimizer (=Adam)
        :param device:      torch.device("cpu" or "cuda")
        :return: None
        """
        self.model = agent_models.PGModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(device)
    
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
    """
    gamma = kwargs.get('gamma', 0.99)
    agent = agents[0]

    #while True:
    for _ in range(2000000):
        episode = generate_episode(env)

        agent.model.optim.zero_grad()
        returns = compute_returns(env, episode, gamma)

        returns_v = torch.tensor(returns).to(agent.device)
        episode = list(zip(*episode)) # reorganise episode
        states_v  = preprocess(episode[0]).to(agent.device)
        actions_v = torch.LongTensor([a[0] for a in episode[1]]).to(agent.device) # TODO: correct for multiple agents

        _, logprob = agent.model(states_v)
        logprob_a = logprob.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss = - torch.sum(logprob_a * returns_v)

        print('Loss = ', loss.item())

        loss.backward()
        agent.model.optim.step()

        """
        states = 
        for Q, (state, actions, _, _, _) in zip(returns, episode):
            action = actions[0] # TODO: correct for multiple agents
            _, logprob = agent.model(preprocess([state]))
            logprob_a = logprob[0, action]
        """