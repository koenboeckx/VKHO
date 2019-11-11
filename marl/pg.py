from . import agent_models
from .common import preprocess
from tensorboardX import SummaryWriter
from collections import namedtuple

# to import agents
import sys
sys.path.insert(1, '/home/koen/Programming/VKHO/game')
import agents

import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter

SAVE_RATE = 1000

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
    
    def save_model(self, filename=None):
        if filename is None:
            filename = './marl/models/pg_agent_{}_{}.torch'.format(self.idx, '01')
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
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
    while env.terminal(state) == 0:
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

def compute_mean_reward(episodes):
    "Compute mean reward for agent 0"
    rewards = 0.0
    n_steps = 0
    for _, _, reward, _, _ in episodes:
        if reward[0] != 0:
            rewards += reward[0]
            n_steps += 1
    return len(episodes)/n_steps, rewards/n_steps

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
    agent = agents[0]

    with SummaryWriter(comment='-pg') as writer:
        for step_idx in range(n_steps):
            episodes = []
            returns  = []
            for _ in range(n_episodes):
                episode = generate_episode(env)
                returns_ = compute_returns(env, episode, gamma)
                episodes.extend(episode)
                returns.extend(returns_)
            
            mean_length, mean_reward = compute_mean_reward(episodes)
            writer.add_scalar('mean_reward', mean_reward, step_idx)
            writer.add_scalar('mean_length', mean_length, step_idx)

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
                logprob_act_vals_v = returns_v * logprob_v[range(n_states), actions_v]
                loss = - logprob_act_vals_v.mean()

                print('Agent {}: Loss = {:.5f}'.format(str(agent), loss.item()))
                writer.add_scalar('loss_{}'.format(str(agent)), loss.item(), step_idx)

                loss.backward()
                agent.model.optim.step()
            if step_idx > 0 and step_idx % SAVE_RATE == 0:
                for agent in agents:
                    agent.save_model()

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