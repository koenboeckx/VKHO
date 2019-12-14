"An example to show how to set up a pommerman game programmatically"
import gym
import pommerman
from pommerman import agents

RENDER = True

class TrainingAgent(agents.BaseAgent):
    def __init__(self, character=pommerman.characters.Bomber):
        super().__init__(character)
    
    def act(self, obs, action_space):
        return action_space.sample()

class PommermanEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs = self.env.get_observations()
        all_actions = [actions] + self.env.act(obs)
        state, reward, done, _ = self.env.step(all_actions)
        agent_state  = self.env.featurize(state[self.env.training_agent])
        agent_reward = reward[self.env.training_agent]
        return agent_state, agent_reward, done, {}

def main():
    """
    Simple function to bootstrap a game.
    """
    # Print all possible environements in the Pommerman registry
    print(pommerman.REGISTRY)

    training_agent = TrainingAgent()
    # Create a set of agents (exactly four)
    agent_list = [
        training_agent,
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    env.set_training_agent(training_agent.agent_id)
    env = PommermanEnvWrapper(env)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(10):
        state = env.reset()
        done = False
        while not done:
            if RENDER: env.render()
            action = training_agent.act(state, env.action_space)
            state, reward, done, info = env.step(action)
        print('Episode {} finished'.format(i_episode))
    env.close()

if __name__ == '__main__':
    main()