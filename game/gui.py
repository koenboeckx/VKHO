"""methods to visualize what's happening in the game"""

import pygame
import time
from envs import all_actions # to display actions
import imageio # for generation of gifs

DEBUG = True
MAKE_GIF = True

STEP = 50 # number of pixels per case
PERIOD = 75 # screen update frequency

# define the RGB value for white, 
#  green, blue colour . 
WHITE = (255, 255, 255) 
GREEN = (0, 255, 0) 
BLUE  = (0, 0, 128)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GRAY  = (128, 128, 128)

def visualize(env, period=None):
    """Takes a game environment as argument and visualizes 
    the different steps of the game on the screen. Period = update delay in seconds"""

    if period is None:
        period = 75 # ms
    else:
        period = int(period * 1000) # in ms

    def show_aiming(state, agent, action):
        start = list(reversed([x*STEP+STEP//3 for x in state.positions[agent.idx]]))
        stop  = list(reversed([x*STEP+STEP//3 for x in state.positions[opponent.idx]]))
        line = pygame.draw.line(screen, WHITE, start, stop)
        pygame.display.flip()

        return line
    
    SCREEN_HEIGHT = STEP * env.board_size
    SCREEN_WIDTH  = STEP * env.board_size

    def screenshot(idx):
        pygame.image.save(screen, f"./game/screenshots/screenshot0{idx}.png")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font('freesansbold.ttf', 12)
    text = font.render(' ', True, WHITE, BLUE)
    text_rect = text.get_rect() 

    # create a custom event for adding a new enemy
    STEPEVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEPEVENT, period) # fire STEPEVENT event every period
    
    # create the initial game state and initialze objects
    state = env.get_init_game_state()
    tanks = []
    for idx in range(4):
        init_pos = state.positions[idx]
        tanks.append(Tank(idx, init_pos))

    running = True
    idx, lines = 0, [None,] * len(env.agents)
    while running:
        for event in pygame.event.get():    # global event handling loop
            if event.type == pygame.QUIT:   # did the user click the window close button?
                running = False
            
            if event.type == STEPEVENT:
                actions = env.get_actions(state)
                parse_actions(actions)
                text = font.render(f'{[all_actions[action] for action in actions]} ', True, WHITE, BLUE)
                
                # draw lines between agents when aiming
                for agent, action in zip(env.agents, actions):
                    if state.alive[agent.idx] == 0:
                        tanks[agent.idx].set_dead()
                    if state.aim[agent.idx]:
                        opponent = env.agents[state.aim[agent.idx]]
                        lines[agent.idx] = show_aiming(state, agent, opponent) # uncomment to see aim lines
                    else:
                        lines[agent.idx] = None

                state = env.step(state, actions)
                for tank_idx, tank in enumerate(tanks):
                    pos = state.positions[tank_idx]
                    tank.update(pos)
                
                if MAKE_GIF:
                    #surf = pygame.Surface() # I'm going to use 100x200 in examples
                    screenshot(idx)
                    idx += 1

            if env.terminal(state):
                if env.terminal(state) == 1:
                    print('Team 0 won!')
                elif env.terminal(state) == -1:
                    print('Team 1 won!')
                running = False

        screen.fill((0, 0, 0))
        for tank in tanks:
            screen.blit(tank.surf, tank.rect)
        screen.blit(text, text_rect)
        #for line in lines: # gives error => TODO: improve
        #    if line is not None:
        #        screen.blit(line)
        
        # update the display
        pygame.display.update()
    
    # generate final screen
    print('game over')
    for agent in env.agents:
        if state.alive[agent.idx] == 0:
            tanks[agent.idx].set_dead()
    for tank in tanks:
        screen.blit(tank.surf, tank.rect)          
    pygame.display.update()
    #input('Press any key to continue...')
    if MAKE_GIF:
        #surf = pygame.Surface() # I'm going to use 100x200 in examples
        screenshot(idx)
        idx += 1

def parse_actions(actions):
    for idx, action in enumerate(actions):
        print('Agent {} does {}'.format(idx, all_actions[action]))

class Tank(pygame.sprite.Sprite):
    def __init__(self, idx, init_pos):
        super(Tank, self).__init__()
        self.idx = idx
        self.font = pygame.font.Font('freesansbold.ttf', 28)

        if self.idx in [0, 1]:
            self.surf = self.font.render('T'+str(self.idx), True, WHITE, RED)
        else:
            self.surf = self.font.render('T'+str(self.idx), True, WHITE, BLUE)
        self.rect = self.surf.get_rect(
            center=(init_pos[1]*STEP + STEP/2, init_pos[0]*STEP)
        )
    
    def update(self, pos):
        self.rect.x, self.rect.y= pos[1]*STEP, pos[0]*STEP
    
    def set_dead(self):
        if self.idx in [0, 1]:
            self.surf = self.font.render('T'+str(self.idx), True, GRAY, RED)
        else:
            self.surf = self.font.render('T'+str(self.idx), True, GRAY, BLUE)

def create_gif(path):
    import imageio, os
    filenames = os.listdir(path)
    images = []
    for filename in sorted(filenames):
        images.append(imageio.imread(path+filename))
    imageio.mimsave(path+'movie.gif', images, 'GIF', duration=1)