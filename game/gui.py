"""methods to visualize what's happening in the game"""

import pygame
import time
from envs import all_actions # to display actions

DEBUG = True

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
        if action == 1:
            opponent = env.agents[2] if agent.idx in [0, 1] else env.agents[0]
        elif action == 2:
            opponent = env.agents[3] if agent.idx in [0, 1] else env.agents[1]
        start = [x*STEP for x in state.positions[agent.idx]]
        stop  = [x*STEP for x in state.positions[opponent.idx]]
        pygame.draw.line(screen, WHITE, start, stop)
        pygame.display.flip()

    SCREEN_HEIGHT = STEP * env.board_size
    SCREEN_WIDTH  = STEP * env.board_size

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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
    while running:
        for event in pygame.event.get():    # global event handling loop
            if event.type == pygame.QUIT:   # did the user click the window close button?
                running = False
            if env.terminal(state):
                if env.terminal(state) == 1:
                    print('Team 0 won!')
                elif env.terminal(state) == -1:
                    print('Team 1 won!')
                running = False
            elif event.type == STEPEVENT:
                actions = env.get_actions(state)
                if DEBUG:
                    print('------------------------') 
                    parse_actions(actions)
                
                # draw lines between agents when aiming
                for agent, action in zip(env.agents, actions):
                    if state.alive[agent.idx] == 0:
                        tanks[agent.idx].set_dead()
                    if action in [1, 2]:
                        show_aiming(state, agent, action)

                state = env.step(state, actions)
                for idx, tank in enumerate(tanks):
                    pos = state.positions[idx]
                    tank.update(pos)
        
        screen.fill((0, 0, 0))
        for tank in tanks:
            screen.blit(tank.surf, tank.rect)
        
        # update the display
        pygame.display.update()

def parse_actions(actions):
    for idx, action in enumerate(actions):
        print('Agent {} does {}'.format(idx, all_actions[action]))

class Tank(pygame.sprite.Sprite):
    def __init__(self, idx, init_pos):
        super(Tank, self).__init__()
        self.idx = idx
        #self.surf = pygame.Surface((STEP, STEP))

        self.font = pygame.font.Font('freesansbold.ttf', 28)
        #self.surf = font.render('T'+str(idx), True, blue, blue)


        if self.idx in [0, 1]:
            self.surf = self.font.render('T'+str(self.idx), True, WHITE, RED)
        else:
            self.surf = self.font.render('T'+str(self.idx), True, WHITE, BLUE)
        self.rect = self.surf.get_rect(
            center=(init_pos[0]*STEP, init_pos[1]*STEP)
        )
    
    def update(self, pos):
        self.rect.x, self.rect.y= pos[0]*STEP, pos[1]*STEP
    
    def set_dead(self):
        if self.idx in [0, 1]:
            self.surf = self.font.render('T'+str(self.idx), True, GRAY, RED)
        else:
            self.surf = self.font.render('T'+str(self.idx), True, GRAY, BLUE)

