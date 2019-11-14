from . import agent_models
from .common import preprocess_extended as preprocess
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
import torch.nn.utils as nn_utils

from tensorboardX import SummaryWriter
import numpy as np

SAVE_RATE = 1000
REWARDS_STEPS = 4 # TODO: look this up
CLIP_GRAD = 0.1

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
        self.model = agent_models.PGExtendedModel(input_shape, n_actions,
            lr=lr, board_size=self.board_size).to(self.device)
    
    def save_model(self, filename=None):
        if filename is None:
            filename = './marl/models/pg_agent_{}_{}.torch'.format(self.idx, '01')
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def get_action(self, state):
        board_v, state_v = [tensor.to(self.device) for tensor in preprocess([state])]
        _, logprobs = self.model(board_v, state_v)
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

def generate_steps(env, n_steps):
    """
    Generate n_steps through interaction with environment env. Actions are
    picked with agent.get_action(state) methods for the agents in env.agents.
    """
    experiences = []
    state = env.get_init_game_state()
    for _ in range(n_steps):
        actions = [agent.get_action(state) for agent in env.agents]
        next_state = env.step(state, actions)
        reward = env.get_reward(next_state)
        done = True if env.terminal(next_state) else False
        experiences.append(Experience(state, actions, reward, next_state, done))
        if done:
            state = env.get_init_game_state() # reinit game when episode is done
        else:
            state = next_state
    return experiences
            
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

def compute_mean_reward(episodes):
    "Compute mean reward for agent 0"
    rewards = 0.0
    n_episodes = 0
    for _, _, reward, _, done in episodes:
        rewards += reward[0]
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

    with SummaryWriter(comment='-pg') as writer:
        #writer.add_graph(agents[0].model) # TODO: doesn't work (yet)
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
                states_v  = [tensor.to(agent.device) for tensor in preprocess(episodes[0])]
                actions_v = torch.LongTensor([a[agent.idx] for a in episodes[1]]).to(agent.device)

                _, logits_v = agent.model(*states_v)
                logprob_v = F.log_softmax(logits_v, dim=1)
                logprob_act_vals_v = returns_v * logprob_v[range(n_states), actions_v]
                loss = - logprob_act_vals_v.mean()

                print('Agent {}: Loss = {:.5f}'.format(str(agent), loss.item()))
                writer.add_scalar('loss_{}'.format(str(agent)), loss.item(), step_idx)

                loss.backward()
                agent.model.optim.step()

            if step_idx > 0 and step_idx % SAVE_RATE == 0:
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                for agent in agents:
                    filename = '/home/koen/Programming/VKHO/marl/models/pg_agent{}_{}.torch'.format(agent.idx, date_str)
                    agent.save_model(filename)

def actor_critic(env, agents, **kwargs):
    """
    Use Actor-Critic learning to train agents. Uses environement env.
    :param env: Environment
    :param agents: list of agents to train (subset of env.agents)

    :return: None
    """
    gamma = kwargs.get('gamma', 0.99)           # discount factor
    n_steps = kwargs.get('n_steps', 128)        # number of steps to generate each iteration
    batch_size = kwargs.get('batch_size', 32)   # number of exp used for learning

    with SummaryWriter(comment='-ac') as writer:
        for step_idx in range(100): # TODO: redefine (number of steps or until convergence)
            exp_source = generate_steps(env, n_steps) # TODO: IDEA: make this a generator (-> yield)
            batch = []

            mean_length, mean_reward = compute_mean_reward(exp_source)
            writer.add_scalar('mean_reward', mean_reward, step_idx)
            writer.add_scalar('mean_length', mean_length, step_idx)

            print('Step {:3d} - mean length = {:.2f}, mean reward = {:.3f}'.format(
                step_idx, mean_length, mean_reward
            ))

            for exp in exp_source:
                batch.append(exp)
                if len(batch) >= batch_size:
                    for agent in agents:
                        (boards_v, states_v), actions_t, vals_ref_v = unpack_batch(env, agent, batch, gamma)
                        
                        agent.model.optim.zero_grad()

                        values_v, logits_v = agent.model(boards_v, states_v)
                        loss_value_v = F.mse_loss(values_v.squeeze(-1), vals_ref_v) # loss for value estimate

                        logprob_v = F.log_softmax(logits_v, dim=1)
                        adv_v = vals_ref_v - values_v.detach()
                        logprob_actions_v = adv_v * logprob_v[range(batch_size), actions_t]
                        loss_policy_v = -logprob_actions_v.mean()   # policy gradient

                        # TODO: add entropy loss
                        
                        loss_policy_v.backward(retain_graph=True)

                        loss_v = loss_value_v
                        loss_v.backward()
                        
                        # clip gradients
                        nn.utils.clip_grad_norm_(agent.model.parameters(),
                                                 CLIP_GRAD)


                        agent.model.optim.step()

                    
                    batch.clear()

def unpack_batch(env, agent, batch, gamma):
    """
    Convert batch into training tensors
    :param env:
    :param agent:
    :param batch:
    :param gamma:
    :return: (boards variable, states variable), actions tensor, reference values variable
    """
    states, actions, rewards, not_done_idx, last_states = [], [], [], [], []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.actions[agent.idx])
        rewards.append(exp.reward[agent.idx])
        if exp.done:
            not_done_idx.append(idx)
            last_states.append(exp.state)
    
    boards_v, states_v = [torch.tensor(tensor).to(agent.device) 
                                for tensor in preprocess(states)]
    actions_t = torch.LongTensor(actions).to(agent.device)

    # handle the rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx: # check if not_done is present 
        last_boards_v, last_states_v = [torch.tensor(tensor).to(agent.device) 
                                            for tensor in preprocess(last_states)]
        last_vals_v, _ = agent.model(last_boards_v, last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0] # TODO: check this 
        rewards_np[not_done_idx] += gamma ** REWARDS_STEPS * last_vals_np
    ref_vals_v = torch.tensor(rewards_np).to(agent.device)

    return (boards_v, states_v), actions_t, ref_vals_v


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