"""
Tools that can be used for visualization.
"""

from pymongo import MongoClient
from matplotlib import pyplot as plt
import pygame
import time
import imageio # for generation of gifs
import numpy as np
import torch

from settings import args

action_names = {
    0: 'do nothing',
    1: 'fire',
    2: 'move north',
    3: 'move south',
    4: 'move west',
    5: 'move east',
    6: 'aim 0',
    7: 'aim 1',

}

def plot(ids, keys, filename=None, show=True):
    #client = MongoClient('localhost', 27017)
    client = MongoClient('ampere.elte.rma.ac.be', 27017)
    db = client["my_database"]
    metrics = db.metrics
    fig, ax = plt.subplots()
    for key in keys:
        for id in ids:
            result = metrics.find_one({'run_id': id, 'name' : key})
            x = result['steps']
            y = result['values']
            ax.plot(x, y, label=key)
            plt.xlabel('episode')
    plt.legend()
    if filename is not None:
        plt.savefig(args.path+filename)
    if show: plt.show()

def plot_window(runs, keys, window_size=100, filename=None, show=True, limit_length=None):
    "Plot values averaged over a window"
    client = MongoClient('localhost', 27017)
    #client = MongoClient('ampere.elte.rma.ac.be', 27017)
    db = client["my_database"]
    metrics = db.metrics
    fig, ax = plt.subplots()
    for key in keys:
        for run in runs:
            result = metrics.find_one({'run_id': run, 'name' : key})
            if result is None:
                continue
            x = np.array(result['steps'])
            y = np.array(result['values'])
            if limit_length is not None:
                x = x[:limit_length//x[0]] # takes into account step size > 1
                y = y[:limit_length//x[0]]
            y_mean = np.array([np.mean(y[idx-window_size:idx]) for idx in range(window_size, len(y))])
            # TODO: remove term "0.1" below - variance should be taken over multiple runs
            y_std  = 0.2*np.array([np.std(y[idx-window_size:idx])  for idx in range(window_size, len(y))])
            x = x[range(len(y_mean))]
            ax.plot(x, y_mean, label=f'{runs[run]} - {key}')
            ax.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.2)

            plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.grid()
    if filename is not None:
        plt.savefig(args.path+filename)
    if show: plt.show()

def visualize(env, episode, period=None):
    """Visualizes an episode
    :param env: instance of Environment
    :param episode: list of Experience(s)
    """

    STEP = 50 # number of pixels per case

    # define the RGB value for white, 
    #  green, blue colour . 
    WHITE = (255, 255, 255) 
    GREEN = (0, 255, 0) 
    BLUE  = (0, 0, 128)
    BLACK = (0, 0, 0)
    RED   = (255, 0, 0)
    GRAY  = (128, 128, 128)

    class Tank(pygame.sprite.Sprite):
        def __init__(self, agent_id, init_pos):
            super(Tank, self).__init__()
            self.id = agent_id
            self.team = "blue" if agent_id in [0, 1] else "red"
            self.font = pygame.font.Font('freesansbold.ttf', 28)

            if self.team == 'blue':
                self.surf = self.font.render('T'+str(self.id), True, WHITE, BLUE)
            elif self.team == 'red':
                self.surf = self.font.render('T'+str(self.id), True, WHITE, RED)
            self.rect = self.surf.get_rect(
                center=(init_pos[1]*STEP + STEP/2, init_pos[0]*STEP)
                #center=(init_pos[0]*STEP + STEP/2, init_pos[1]*STEP)
            )
        
        def update(self, pos):
            self.rect.x, self.rect.y= pos[1]*STEP, pos[0]*STEP
            #self.rect.x, self.rect.y= pos[0]*STEP, pos[1]*STEP
        
        def set_dead(self):
            if self.team == 'blue':
                self.surf = self.font.render('T'+str(self.id), True, GRAY, BLUE)
            elif self.team == 'red':
                self.surf = self.font.render('T'+str(self.id), True, GRAY, RED)
    
    class Obstacle(pygame.sprite.Sprite):
        def __init__(self, position):
            super(Obstacle, self).__init__()
            self.position = position
            x, y = position
            self.rect = pygame.rect.Rect((x*STEP, y*STEP, STEP, STEP))
        
        def draw(self):
            pygame.draw.rect(screen, BLACK, self.rect)


    if period is None:
        period = 75 # ms
    else:
        period = int(period * 1000) # in ms

    def show_aiming(state, agent, opponent):
        start = list(reversed([x*STEP+STEP//3 for x in state[agent,    :2]]))
        stop  = list(reversed([x*STEP+STEP//3 for x in state[opponent, :2]]))
        line = pygame.draw.line(screen, BLACK, start, stop)
        pygame.display.flip()

        return line
    
    SCREEN_HEIGHT = STEP * env.board_size
    SCREEN_WIDTH  = STEP * env.board_size

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font('freesansbold.ttf', 12)
    text = font.render(' ', True, WHITE, BLUE)
    text_rect = text.get_rect() 

    # create a custom event for adding a new enemy
    STEPEVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEPEVENT, period) # fire STEPEVENT event every period
    
    # get the initial game state and initialze objects
    state = episode[0].state
    n_agents = state.size(0) # n_rows == n_agents
    tanks = []
    for agent_idx in range(n_agents):
        init_pos = state[agent_idx, :2]
        tanks.append(Tank(agent_idx, init_pos))
    
    # create terrain
    obstacles = []
    for x, y in env.terrain:
        obstacles.append(Obstacle((x, y)))

    running = True
    idx, lines = 0, [None,] * len(env.agents)
    while running:
        for event in pygame.event.get():    # global event handling loop
            if event.type == pygame.QUIT:   # did the user click the window close button?
                running = False
            
            if event.type == STEPEVENT:
                idx += 1
                if idx == len(episode) - 1:
                    running = False
                
                exp = episode[idx]
                state = exp.state
                #agents = exp.state.agents
                actions = [int(a) for a in exp.actions]

                text = font.render(f'{[action_names[action] for action in actions]} ', True, WHITE, BLUE)
                
                # draw lines between agents when aiming
                for agent_id in range(n_agents):
                    if state[agent_id, 2] == 0:
                        tanks[agent_id].set_dead()
                    if state[agent_id, 4] > -1:
                        opponent = state[agent_id, 4]
                        lines[agent_id] = show_aiming(state, agent_id, int(opponent.item())) # uncomment to see aim lines
                    else:
                        lines[agent_id] = None

                for tank in tanks:
                    pos = state[tank.id, :2]
                    tank.update(pos)
                
        screen.fill(WHITE)
        for tank in tanks:
            screen.blit(tank.surf, tank.rect)
        for obstacle in obstacles:
            obstacle.draw()
        #    screen.blit(obstacle.surf, obstacle.rect) 
        screen.blit(text, text_rect)
        
        # update the display
        pygame.display.update()
        pygame.image.save(screen, f"{args.path}screenshot0{idx}.png")
    
    # generate final screen
    print('game over')
    for agent_id in range(n_agents):
        if exp.state[agent_id, 2] == 0:
            tanks[agent_id].set_dead()
    for tank in tanks:
        screen.blit(tank.surf, tank.rect)
     
    pygame.display.update()
    input('Press any key to continue...')

def create_gif(path):
    import imageio, os
    filenames = os.listdir(path)
    images = []
    for filename in sorted(filenames):
        images.append(imageio.imread(path+filename))
    imageio.mimsave(path+'movie.gif', images, 'GIF', duration=1)

def test_run():
    #plot(ids=[672, 675], keys=['length', 'reward'], filename='test')
    #plot_window(ids=[672, 675], keys=['reward', 'win_blue'], filename='test', window_size=200)

    runs = {
        #706:    'QMIX',
        #707:    'VDN',
        713:    'REINFORCE'
    }
    runs = {31: 'IQL'}
    runs = {396: 'PG', 419: 'QMIX'}
    plot_window(runs=runs, keys=['reward'], filename='pg_v_qmix_simple', window_size=200, limit_length=21000)

def test_replay(model_file, mixer_file=None):
    import yaml
    from utilities import get_args
    args = get_args(yaml.load(open('default_config.yaml', 'r')))
    path = '/home/koen' + args.path
    #from utilities import generate_episode
    from env import RestrictedEnvironment, Agent
    from qmix import QMIXAgent, generate_models, generate_episode

    models = generate_models(args.n_inputs, args.n_actions, args)
    models['model'].load_state_dict(torch.load(path+model_file))

    team_blue = [QMIXAgent(idx, "blue", args) for idx in range(args.n_friends)]
    for agent in team_blue:
        agent.set_model(models)
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]
    agents = team_blue + team_red
    env = RestrictedEnvironment(agents, args)
    episode = generate_episode(env, args)
    print(len(episode))
    visualize(env, episode)

if __name__ == '__main__':
    test_replay('RUN_428_MODEL.torch', mixer_file='RUN_428_MIXER.torch')