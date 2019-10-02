"""methods to visualize what's happening in the game"""

import pygame
from game.envs import all_actions # to display actions

DEBUG = True

STEP = 50 # number of pixels per case

# define the RGB value for white, 
#  green, blue colour . 
WHITE = (255, 255, 255) 
GREEN = (0, 255, 0) 
BLUE  = (0, 0, 128)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)

def visualize(env):
    """Takes a game environment as argument and visualizes 
    the different steps of the game on the screen"""

    SCREEN_HEIGHT = STEP * env.board_size
    SCREEN_WIDTH  = STEP * env.board_size

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # create a custom event for adding a new enemy
    STEPEVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEPEVENT, 100) # fire STEPEVENT event every 2000 ms
    
    # create the initial game state and initialze objects
    obs = env.set_init_game_state()
    tanks = []
    for idx in range(4):
        init_pos = obs[idx].position
        tanks.append(Tank(idx, init_pos))

    running = True
    while running:
        for event in pygame.event.get():    # global event handling loop
            if event.type == pygame.QUIT:   # did the user click the window close button?
                running = False
            elif event.type == STEPEVENT:
                actions = env.act(obs)
                if DEBUG:
                    print('------------------------') 
                    parse_actions(actions)
                
                # draw lines between agents when aiming
                for agent, action in zip(env.agents, actions):
                    if action == 1:
                        if agent.idx in [0, 1]:
                            opponent = env.agents[2]
                        else:
                            opponent = env.agents[0]
                        start = [x*STEP for x in agent.pos]
                        stop  = [x*STEP for x in opponent.pos]
                        pygame.draw.line(screen, WHITE, start, stop)
                        pygame.display.flip()

                obs = env.step(actions)
                for idx, tank in enumerate(tanks):
                    pos = obs[idx].position
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

        font = pygame.font.Font('freesansbold.ttf', 28) 
        #self.surf = font.render('T'+str(idx), True, blue, blue)


        if idx in [0, 1]:
            self.surf = font.render('T'+str(idx), True, WHITE, RED)
        else:
            self.surf = font.render('T'+str(idx), True, WHITE, BLUE)
        self.rect = self.surf.get_rect(
            center=(init_pos[0]*STEP, init_pos[1]*STEP)
        )
    
    def update(self, pos):
        self.rect.x, self.rect.y= pos[0]*STEP, pos[1]*STEP

