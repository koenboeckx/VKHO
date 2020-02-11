from collections import namedtuple

from simple_env import *

params = {
    'board_size':   5,
    'init_ammo':    5,
    'max_range':    3,
    'step_penalty': 0.001,
}

Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations'
])


def generate_episode(env):
    episode = []
    state, done = env.reset(), False
    while not done:
        observations = dict([(agent, env.get_observation(agent)) for agent in env.agents])
        actions = dict([(agent, agent.act(obs)) for agent, obs in zip(env.agents, observations)])
        next_state, rewards, done, _ = env.step(actions)
        episode.append(Experience(state, actions, rewards, next_state, done, observations))
        state = next_state
    return episode



#---------------------------------- test -------------------------------------
def test_generate_episode():
    agents = [Agent(0), Agent(1)]
    env = SimpleEnvironment(agents)
    episode = generate_episode(env)
    print(len(episode))

if __name__ == '__main__':
    test_generate_episode()