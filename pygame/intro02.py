"""From https://realpython.com/pygame-a-primer/"""

import random, os

print("os.pwd() = {}".format(os.getcwd()))

import pygame
from pygame.locals import (
    RLEACCEL,
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
        self.surf = pygame.image.load("pygame/jet.png").convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
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

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super(Enemy, self).__init__()
        self.surf = pygame.image.load("pygame/missile.png").convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        # starting position is randomly generated
        self.rect = self.surf.get_rect(
            center = (
                random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
                random.randint(0, SCREEN_HEIGHT),
            )
        )
        self.speed = random.randint(5, 20)
    
    # Move the sprite based on speed
    # Remove the sprite when it passes the left edge of the screen
    def update(self):
        self.rect.move_ip(-self.speed, 0)
        if self.rect.right < 0:
            self.kill()

# Initialize pygame
pygame.init()

# create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# create a custom event for adding a new enemy
ADDENEMY = pygame.USEREVENT + 1
pygame.time.set_timer(ADDENEMY, 250) # fire ADDENEMY event every 250 ms

# instantiate the player
player = Player()

# create groups to hold enemy sprites and all sprites
#   - enemies is used for collision detection and position updates
#   - all_sprites is used for rendering
enemies = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
all_sprites.add(player)

running = True
while running:
    for event in pygame.event.get():    # global event handling loop
        if event.type == KEYDOWN:       # did the user hit a key?
            if event.type == K_ESCAPE:  # was it the escape key?
                running = False
        elif event.type == QUIT:        # did the user click the window close button?
            running = False

        # add a new enemy?
        elif event.type == ADDENEMY:
            # create the new enemy and add it to sprite groups
            new_enemy = Enemy()
            enemies.add(new_enemy)
            all_sprites.add(new_enemy)

    # get all the keys currently pressed
    pressed_keys = pygame.key.get_pressed()

    # update the player sprite based on user keypresses
    player.update(pressed_keys)

    # update enemies positions
    enemies.update()
    
    # Fill the screen with black
    screen.fill((0, 0, 0))

    # draw all sprites
    for entity in all_sprites:
        screen.blit(entity.surf, entity.rect)
    
    # check if any enemies have collided with the player
    if pygame.sprite.spritecollideany(player, enemies):
        # if so, then remove the player and stop the loop
        player.kill()
        running = False

    # update the display
    pygame.display.flip()