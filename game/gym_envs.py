import gym

class Environment:
    """A wrapper for the GYM environment"""
    def __init__(self, agents, **kwargs):
        self.environment = gym.make('CartPole-v0')
        self.agents = agents
        self.n_actions = self.environment.action_space.n
        self.state_space = self.environment.observation_space.shape
        self.done = False

    def get_init_game_state(self):
        state = self.environment.reset()
        self.done = False
        return state
    
    def render(self, state):
        self.environment.render()
    
    def check_conditions(self, state, agent, action):
        return True
    
    def step(self, state, actions):
        action = actions[0]
        state, reward, done, _ = self.environment.step(action)
        self.reward = reward
        self.done   = done
        return state
    
    def terminal(self, state):
        return self.done
    
    def get_reward(self, state):
        return (self.reward, self.reward,
                -self.reward, -self.reward)