"""
Tools that can be used for visualization.
"""

from pymongo import MongoClient
from matplotlib import pyplot as plt
import pygame
import time
import imageio # for generation of gifs

from settings import args

def plot(id, keys, filename=None, show=True):
    client = MongoClient('localhost', 27017)
    db = client["my_database"]
    metrics = db.metrics
    fig, ax = plt.subplots()
    for key in keys:
        result = metrics.find_one({'run_id': id, 'name' : key})
        x = result['steps']
        y = result['values']
        ax.plot(x, y, label=key)
        plt.xlabel('episode')
    plt.legend()
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
        def __init__(self, agent, init_pos):
            super(Tank, self).__init__()
            self.id = agent.id
            self.team = agent.team
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

    if period is None:
        period = 75 # ms
    else:
        period = int(period * 1000) # in ms

    def show_aiming(state, agent, action):
        start = list(reversed([x*STEP+STEP//3 for x in state.position[agent]]))
        stop  = list(reversed([x*STEP+STEP//3 for x in state.position[opponent]]))
        line = pygame.draw.line(screen, WHITE, start, stop)
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
    tanks = []
    for agent in state.agents:
        init_pos = state.position[agent]
        tanks.append(Tank(agent, init_pos))

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
                agents = exp.state.agents
                actions = list((exp.actions.values()))

                text = font.render(f'{[action.name for action in actions]} ', True, WHITE, BLUE)
                
                # draw lines between agents when aiming
                for agent, action in zip(agents, actions):
                    if state.alive[agent] == 0:
                        tanks[agent.id].set_dead()
                    if state.aim[agent] is not None:
                        opponent = state.aim[agent]
                        lines[agent.id] = show_aiming(state, agent, opponent) # uncomment to see aim lines
                    else:
                        lines[agent.id] = None

                for tank in tanks:
                    agent = agents[tank.id]
                    pos = state.position[agent]
                    tank.update(pos)

        screen.fill(WHITE)
        for tank in tanks:
            screen.blit(tank.surf, tank.rect)
        screen.blit(text, text_rect)
        
        # update the display
        pygame.display.update()
        pygame.image.save(screen, f"{args.path}screenshot0{idx}.png")
    
    # generate final screen
    print('game over')
    for agent in exp.state.agents:
        if exp.state.alive[agent] == 0:
            tanks[agent.id].set_dead()
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

if __name__ == '__main__':
    #plot(id=681, keys=['length', 'reward'], filename='test')

    from utilities import generate_episode
    from env import Environment, Agent

    team_blue = [Agent(idx, "blue") for idx in range(args.n_friends)]
    team_red  = [Agent(args.n_friends + idx, "red") for idx in range(args.n_enemies)]
    agents = team_blue + team_red
    env = Environment(agents)
    episode = generate_episode(env)
    print(len(episode))
    visualize(env, episode)