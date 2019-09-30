from . import agents
import random

class Environment:
    """The base game environment."""
    def __init__(self, **kwargs):
        self.agents = []
        self.n_agents = 0
        self.args = kwargs
        self.state = None

        self.set_init_game_state()
    
    def add_agent(self, id_, agent):
        assert isinstance(agent, agents.BaseAgent)
        self.agents.append(agent)
        self.n_agents += 1
    
    def set_init_game_state(self):
        """Set the initial game state.
        A game state consists of a list of agents states.
        Each agent state consists of
            * ...
        The initial game state is created ....

        Returns: a list of agent states.
        """
        self.board_size = self.args.get('size', 5)
        self.board = {}
        for i in range(self.board_size):
           for j in range(self.board_size):
               self.board[(i,j)] = None
        
        # Position players randomly on the board
        for agent in self.agents:
            i, j = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            agent.pos = (i, j)
            self.board[(i,j)] = repr(agent)


    def render(self):
        """Represent the state of the environment"""
        board_repr = ''
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[(i,j)] == None:
                   board_repr += ' ' * 4
                else:
                    board_repr += ' ' + self.board[(i,j)] + ' '
            board_repr += '\n'
        
        print(board_repr)
        return board_repr

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

    def step(self, actions):
        """Perform actions, part of joint action space.
        Deconflict simultanuous execution of actions (...)
        """
        for agent, action in zip(self.agents, actions):
            pass # change environemnt state based on agent action
