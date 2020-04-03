import copy

import torch

from pg2 import PGAgent, RNNModel
from env import RestrictedEnvironment, Agent
from utilities import get_args
from new_utilities import generate_episode

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment(f'ITER')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))
ex.add_config('default_config.yaml')    # requires PyYAML

def train_agents(env, training_agents, args):
    epi_len, nwins = 0, 0
    n_episodes = 0
    for step_idx in range(int(args.n_steps/args.n_episodes_per_step)):
        batch = []
        for _ in range(args.n_episodes_per_step):
            episode = generate_episode(env, args)
            n_episodes += 1 
            batch.extend(episode)

            epi_len += len(episode)
            reward = episode[-1].rewards["blue"]

            ex.log_scalar('length', len(episode))
            ex.log_scalar('reward', reward)
            ex.log_scalar(f'win_blue', int(episode[-1].rewards["blue"] == 1))
            ex.log_scalar(f'win_red', int(episode[-1].rewards["red"] == 1))

            if episode[-1].rewards["blue"] == 1:
                nwins += 1

        for agent in training_agents:
            loss = agent.update(batch)
            ex.log_scalar(f'loss{agent.id}', loss['loss'])

        s  = f"Step {step_idx}: "
        s += f"Average length: {epi_len/args.n_episodes_per_step:5.2f} - "
        s += f"win ratio: {nwins/args.n_episodes_per_step:4.3f} - "
        print(s)
        epi_len, nwins = 0, 0

    return training_agents

def train_iteratively(args, agent_type):
    # setup the agents & environment
    args.n_actions = 6 + args.n_enemies
    args.n_inputs  = 4 + 3*(args.n_friends-1) + 3*args.n_enemies + args.n_enemies
    # setup model
    if agent_type == 'reinforce':
        model = RNNModel(input_shape=args.n_inputs, n_actions=args.n_actions, args=args)
        team_blue = [PGAgent(idx, "blue", args) for idx in range(args.n_friends)]
    elif agent_type == 'qmix':
        pass
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]

    training_agents = team_blue

    agents = team_blue + team_red
    env = RestrictedEnvironment(agents, args)

    for agent in training_agents:
        agent.set_model(model)

    # first model
    training_agents = train_agents(env, training_agents, args)
    trained_model   = copy.deepcopy(training_agents[0].model)
    
    for iteration in range(args.n_iterations):
        print(f'Iteration {iteration + 1}')
        #args.n_steps = 10000 * (iteration + 2) # adapt step size TODO: find optimal criterion, e.g. stop at certain win rate
        # upgrade team red
        if agent_type == 'reinforce':
            team_red  = [PGAgent(args.n_friends + idx, "red", args) for idx in range(args.n_enemies)]

        training_agents = team_blue

        agents = team_blue + team_red
        env = RestrictedEnvironment(agents, args)

        opponent_model = RNNModel(input_shape=args.n_inputs, n_actions=args.n_actions,
                                  args=args)
        opponent_model.load_state_dict(trained_model.state_dict())
        opponent_model.eval()
        for agent in team_red:
            agent.set_model(opponent_model)
        
        if args.reset_model:
            model = RNNModel(input_shape=args.n_inputs, n_actions=args.n_actions,
                            args=args)
            for agent in team_blue:
                agent.set_model(model)
        
        training_agents = train_agents(env, training_agents, args)
        trained_model = copy.deepcopy(training_agents[0].model)
    
    from os.path import expanduser
    home = expanduser("~")
    torch.save(trained_model.state_dict(), home+args.path+f'RUN_{get_run_id()}_MODEL.torch')

@ex.capture
def get_run_id(_run): # enables saving model with run id
    return _run._id

@ex.automain
def run(_config):
    args = get_args(_config)
    train_iteratively(args, agent_type='reinforce')