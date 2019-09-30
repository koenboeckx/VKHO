from . import agents

class Environment:
    """The base game environment."""
    def __init__(self):
        self.agents = []
    
    def add_agent(self, id_, agent):
        assert isinstance(agent, agents.BaseAgent)
        self.agents.append(agent)
    
    def set_init_game_state(self):
        """Set the initial game state.
        A game state consists of a list of agents states.
        Each agent state consists of
            * ...
        The initial game state is created ....

        Returns: a list of agent states.
        """
        pass

    def act(self, observations):
        """Return the chosen action for eac agent,
        based on the global observation.
        :params:
            observations: tuple oof individual observations
        :returns:
            list of actions, one for each agent in agent_list
        """
        actions = []
        for obs, agent in zip(observations, self.agents):
            actions.append(agent.get_action(obs))
        return actions

    def step(self, obs, actions):
        """Perform actions, part of joint action space.
        Deconflict simultanuous execution of actions (...)
        """
        for agent, action in zip(self.agents, actions):
            pass # change environemnt state based on agent action