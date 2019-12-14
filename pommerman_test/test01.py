"""
Based on https://github.com/rwightman/pytorch-pommerman-rl
-- Single-Agent RL --
"""

import random

import numpy as np
import gym
import pommerman
from pommerman import agents
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    """Wrapper for neural network `nn` that samples from 
    actor_features"""
    def __init__(self, nn, action_space):
        super().__init__()
        self.nn = nn
    
    def act(self, inputs, hidden, masks):
        """Choose action from action space, based on logits that result form
        applying inputs to self.nn(inputs, masks)
        :param inputs:
        :param masks:
        :return: value, action log_probs
        """
        value, logits, rnn_hxs = self.nn(inputs, hidden, masks)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return value, action, log_probs, rnn_hxs
    
    def get_value(self, inputs, hidden, masks):
        value, _, _ = self.nn(inputs, hidden, masks)
        return value
    
    def evaluate_actions(self, inputs, hidden, masks, actions):
        value, logits, rnn_hxs = self.nn(inputs, hidden, masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze()) # FIXME: can give problem when num_processes > 1
        dist_entropy = dist.entropy().mean()
        return value, log_probs, dist_entropy, rnn_hxs

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.value = nn.Linear(hidden_size, 1)
        self.logits = nn.Linear(hidden_size, n_actions)
    
    def forward(self, inputs, *args):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        logits = self.logits(x)
        return value, logits, None

def create_policy(obs_shape, action_space, name='basic', nn_kwargs={}, train=True):
    if name == 'basic':
        nn = BaseModel(obs_shape[0], **nn_kwargs)
    
    if train: nn.train() # has only effect on Dropout, BatchNorm, ...
    else:     nn.eval()

    policy = Policy(nn, action_space=action_space)
    return policy

class A2C_Agent():
    def __init__(self, actor_critic, value_loss_coef, entropy_coef,
                 lr, eps, alpha, max_grad_norm):
        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr=lr,
                                        eps=eps, alpha=alpha)
    
    def update(self, rollouts, update_index, replay=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1] # always gives 1 ?
        num_steps, num_processes, _ = rollouts.rewards.size()
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            None, # dummy for hidden states
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape)
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        loss = value_loss * self.value_loss_coef + action_loss + dist_entropy * self.entropy_coef
        loss.backward()

        nn.utils.clip_grad_norm(self.actor_critic.parameters(),
                                self.max_grad_norm)
        return value_loss.item(), action_loss.item(), dist_entropy.item()

class RolloutStorage:
    """Store all information (observations, actions, ...) directly in torch tensors.
    Only possible because number of steps `num_steps` is known a priori."""
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape) # +1 for initial state
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
    
    def to(self, device):
        "move all tensor to device"
        raise NotImplementedError # TODO: implement this
    
    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs) # alternative to self.obs[self.step + 1] = obs - avoids copying??
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps
    
    def after_update(self):
        """Call fter update of the agent; move last value to first position where it matters (obs, masks);
        This allows to reuse the RolloutStorage in next update."""
        self.obs[0].copy_(self.obs[-1])
        self.rewards[0].copy_(self.rewards[-1])
    
    def compute_returns(self, next_value, gamma, tau = 0.0):
        """Compute returns at the end of sequence to perform update.
        :param next_value:  the predicted value after num_steps
        :param gamma:       discount factor
        :param tau:         [only used for GAE] # TODO: Implement this
        """
        self.returns[-1] = next_value # next_value is the predicted
        for step in reversed(range(self.num_steps)):
            self.returns[step] = gamma * self.masks[step + 1] * \
                self.returns[step + 1] + self.rewards[step]

class TrainingAgent(agents.BaseAgent):
    def __init__(self, character=pommerman.characters.Bomber):
        super().__init__(character)
    
    def act(self, obs, action_space):
        return action_space.sample()

class PommermanEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.item() # adapt for multiple actions (i.e. multiple learning agents)
        obs = self.env.get_observations()
        all_actions = [actions] + self.env.act(obs)
        state, reward, done, _ = self.env.step(all_actions)
        agent_state  = self.env.featurize(state[self.env.training_agent])
        agent_reward = reward[self.env.training_agent]
        return agent_state, agent_reward, done, {}
    
    def reset(self):
        obs = self.env.reset()
        agent_obs = self.env.featurize(obs[self.env.training_agent]) 
        return agent_obs

class ListEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
    
    def step(self, action):
        num_processes = len(self.envs)
        env = self.envs[0] # FIXME: should be applicable for multiple environments
        obs, reward, done, info = env.step(action)
        obs = torch.tensor(obs)
        reward = torch.tensor(reward)
        return obs, reward, [done], info
    
    def reset(self):
        return torch.tensor([env.reset() for env in self.envs])

def make_vec_env(num_processes):
    return ListEnvWrapper([make_env() for _ in range(num_processes)])

def make_env():
    training_agent = TrainingAgent()
    # Create a set of agents (exactly four)
    agent_list = [
        training_agent,
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    env.set_training_agent(training_agent.agent_id)
    return PommermanEnvWrapper(env)

class Arguments:
    num_updates = 20
    num_steps = 10
    num_processes = 1
    obs_shape = (372,)
    action_space = 5 
    nn_kwargs = {
        'hidden_size': 512,
        'n_actions': action_space
    }
    lr = 0.001
    eps = 0.1
    alpha = 0.1 # what is this? to change
    max_grad_norm = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = 0.99
    tau = None # not important now; change later
    poliycy_name = 'basic'
    value_loss_coef = 1.0
    entropy_coef = 1.0

def main(args):
    train_envs = make_vec_env(args.num_processes)
    obs = train_envs.reset()
    #args.obs_space = 
    actor_critic = create_policy(args.obs_shape, args.action_space,
                                name=args.poliycy_name, nn_kwargs=args.nn_kwargs)
    agent = A2C_Agent(actor_critic, args.value_loss_coef, args.entropy_coef,
                      args.lr, args.eps, args.alpha, args.max_grad_norm)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              args.obs_shape, args.action_space)

    obs = train_envs.reset()
    rollouts.obs[0].copy_(obs)

    for j in range(args.num_updates):
        print(f'Update {j}')
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, log_probs, rnn_hxs = actor_critic.act(
                    rollouts.obs[step], None, # dummy for hidden state
                    rollouts.masks[step])
            obs, reward, done, infos = train_envs.step(action)
            masks = torch.tensor([[0.0] if done_ else [1.0]
                                    for done_ in done], device=args.device)
            rollouts.insert(obs, action, log_probs, value, reward, masks)
        
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                None, # dummy for hidden state
                                                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.gamma, args.tau) 
        value_loss, action_loss, dist_entropy = agent.update(rollouts, j)
        rollouts.after_update()                                            


if __name__ == '__main__':
    args = Arguments()
    main(args)