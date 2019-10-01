import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

STEP = 2
SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600

# define a player object by extending pygame.sprite.Sprite
# the surface drawn on the screen is now an attribute of player
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((75, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
    
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -STEP)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0,  STEP)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-STEP, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(STEP,  0)

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

# Initialize pygame
pygame.init()

# create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# screen.fill((255, 255, 255)) # fill -> background. White (255, 255, 255) here

# instantiate the player
player = Player()

running = True
while running:
    for event in pygame.event.get():
        if event.type == KEYDOWN:       # did the user hit a key?
            if event.type == K_ESCAPE:  # was it the escape key?
                running = False
        elif event.type == QUIT:        # did the user click the window close button?
            running = False

    # get all the keys currently pressed
    pressed_keys = pygame.key.get_pressed()

    # update the player sprite based on user keypresses
    player.update(pressed_keys)
    
    # Fill the screen with black
    screen.fill((0, 0, 0))

    # draw the player on the screen
    #screen.blit(player.surf, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
    screen.blit(player.surf, player.rect)

    # update the display
    pygame.display.flip()