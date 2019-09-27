"""Entry point to the game module"""
import envs
from . import agents

def make(config_id, agent_list, render_mode='human'):
    """Creates the game environment"""
    env = envs.Environment()

    for id_, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        env.add_agent(id_, agent)
