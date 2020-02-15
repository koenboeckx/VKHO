import torch
from torch import nn
from torch.nn import functional as F

from env import *
from utilities import LinearScheduler, ReplayBuffer, Experience, generate_episode
from models import IQLModel
from settings import params

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('QL-2v2')
ex.observers.append(MongoObserver(url='localhost',
                                db_name='my_database'))

class IQLAgent(Agent):
    def __init__(self, id, team, models, params):
        super().__init__(id, team, params)
        self.model  = models['model']
        self.target = models['target']
        self.sync_models()
        self.scheduler = params['scheduler'](start=1.0, stop=0.1, steps=params['scheduler_steps'])
    
    def sync_models(self):
        self.target.load_state_dict(self.model.state_dict())
    
    def act(self, obs):
        if not obs.alive: # if not alive, do nothing
            return 0

        unavail_actions = self.env.get_unavailable_actions()[self]
        avail_actions = [action for action in range(self.env.n_actions)
                            if action not in unavail_actions]
        
        eps = self.scheduler()
        if random.random() < eps:
            return random.choice(avail_actions)
        else:
            with torch.no_grad():
                qvals = self.model([obs])
                # remove unavailable actions
                for action in unavail_actions:
                    qvals[0][action] = -np.infty
                action = qvals.max(1)[1].item()
                return action
    
    def update(self, batch):
        _, actions, rewards, _, dones, observations, next_obs, unavailable_actions = zip(*batch)
        actions = [action[self] for action in actions]
        rewards = torch.tensor([reward[self] for reward in rewards])
        observations = [obs[self] for obs in observations]
        next_obs = [obs[self] for obs in next_obs]
        dones = torch.tensor(dones, dtype=torch.float)
        
        current_qvals = self.model(observations)[range(len(batch)), actions]
        
        predicted_qvals = self.target(next_obs)
        # Set unavailable action to very low Q-value !!
        for idx in range(len(unavailable_actions)):
            unavail_actions = unavailable_actions[idx][self]
            predicted_qvals[idx][unavail_actions] = -np.infty

        predicted_qvals_max = predicted_qvals.max(1)[0]
        targets = rewards + params['gamma']*(1.-dones) * predicted_qvals_max

        self.model.optimizer.zero_grad()
        loss = F.mse_loss(current_qvals, targets.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), params['clip'])
        self.model.optimizer.step()
        return loss.item()

def generate_models():
    model  = IQLModel(input_shape=13, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    target = IQLModel(input_shape=13, n_hidden=params['n_hidden'], n_actions=len(all_actions), lr=params['lr'])
    return {"model": model, "target": target}


#---------------------------------- test -------------------------------------

PRINT_INTERVAL = 100
RENDER = False

@ex.config
def cfg():
    params=params

@ex.automain
def run(params):
    models = generate_models()
    team_blue = [IQLAgent(0, "blue", models, params), IQLAgent(1, "blue", models, params)]
    team_red  = [Agent(2, "red", params),  Agent(3, "red", params)]

    agents = team_blue + team_red
    training_agents = team_blue
    env = Environment(agents, params)
    buffer = ReplayBuffer(size=params['buffer_size'])
    epi_len, nwins = 0, 0
    for step_idx in range(params['n_steps']):
        episode = generate_episode(env)
        buffer.insert_list(episode)
        if not buffer.can_sample(params['batch_size']):
            continue
        epi_len += len(episode)
        if episode[-1].rewards[agents[0]] == 1:
            nwins += 1
        batch = buffer.sample(params['batch_size'])
        for agent in training_agents:
            # TODO: how to handle update on dead agents?
            loss = agent.update(batch)
            if step_idx > 0 and step_idx % params['sync_interval'] == 0:
                agent.sync_models()

            ex.log_scalar(f'loss{agent.id}', loss, step=step_idx)
            ex.log_scalar(f'epsilon', agent.scheduler(), step=step_idx)

        if step_idx > 0 and step_idx % PRINT_INTERVAL == 0:
            s  = f"Step {step_idx}: loss: {loss:8.4f} - "
            s += f"Average length: {epi_len/PRINT_INTERVAL:5.2f} - "
            s += f"win ratio: {nwins/PRINT_INTERVAL:4.3f} - "
            s += f"epsilon: {agent.scheduler():4.3f} - "
            print(s)
            epi_len, nwins = 0, 0
            #_ = generate_episode(env, render=True)

        ex.log_scalar(f'length', len(episode), step=step_idx)
        ex.log_scalar(f'win', int(episode[-1].rewards[agents[0]] == 1), step=step_idx)
    
    for agent in training_agents:
        agent.save(f'IQL-2v2_7_agent{agent.id}.p')
    torch.save(models["model"].state_dict(), 'IQL-2v2_7x7.torch')
    
#if __name__ == '__main__':
#    test_take_action()