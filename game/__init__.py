"""Entry point to the game module"""
from . import envs
from . import agents

def make(config_id, agent_list, render_mode='human', board_size=11):
    """Creates the game environment"""
    env = envs.Environment(board_size=board_size)

    for id_, agent in enumerate(agent_list):
        #assert isinstance(agent, agents.BaseAgent)
        env.add_agent(id_, agent)
    
    return env
