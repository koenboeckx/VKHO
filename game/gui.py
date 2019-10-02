"""methods to visualize what's happening in the game"""

import pygame

STEP = 50 # number of pixels per case

def visualize(env):
    """Takes a game environment as argument and visualizes 
    the different steps of the game on the screen"""

    SCREEN_HEIGHT = STEP * env.board_size
    SCREEN_WIDTH  = STEP * env.board_size

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # create a custom event for adding a new enemy
    STEPEVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEPEVENT, 2000) # fire STEPEVENT event every 2000 ms
    
    # create the initial game state and initialze objects
    obs = env.set_init_game_state()
    tanks = []
    for idx in range(4):
        init_pos = obs[idx].position
        tanks.append(Tank(init_pos))

    running = True
    while running:
        for event in pygame.event.get():    # global event handling loop
            if event.type == pygame.QUIT:   # did the user click the window close button?
                running = False
            elif event.type == STEPEVENT:
                actions = env.act(obs)
                obs = env.step(actions)
                for idx, tank in enumerate(tanks):
                    pos = obs[idx].position
                    tank.update(pos)
        
        screen.fill((0, 0, 0))
        for tank in tanks:
            screen.blit(tank.surf, tank.rect)
        
        # update the display
        pygame.display.flip()


class Tank(pygame.sprite.Sprite):
    def __init__(self, init_pos):
        super(Tank, self).__init__()
        self.surf = pygame.Surface((20, 10))
        self.surf.fill((255, 0, 255))
        self.rect = self.surf.get_rect(
            center=(init_pos[0]*STEP, init_pos[1]*STEP)
        )
    
    def update(self, pos):
        self.rect.move((pos[0]*STEP, pos[1]*STEP))

