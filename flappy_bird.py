import neat
import pygame
import time
import os
import random
pygame.font.init()

window_width = 500
window_height = 800

bird_images = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))
]
pipe_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
base_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
bg_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

Stat_font = os.path.join("C:\\r\\projects\\flappy bird\\imgs", "FontsFree-Net-04B_19__.TTF")
font = pygame.font.Font(Stat_font, 50)

class Bird:
    images = bird_images
    max_rotation = 25
    rotation_velocity = 20
    animation_time = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.images_count = 0
        self.image = self.images[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.velocity * self.tick_count + 1.5 * self.tick_count ** 2

        if displacement >= 16:
            displacement = 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement > 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        else:
            if self.tilt >= -90:
                self.tilt = self.rotation_velocity

    def draw(self, win):
        self.images_count += 1
        if self.images_count < self.animation_time:
            self.image = self.images[0]
        elif self.images_count < self.animation_time * 2:
            self.image = self.images[1]
        elif self.images_count < self.animation_time * 3:
            self.image = self.images[2]
        elif self.images_count < self.animation_time * 4:
            self.image = self.images[1]
        elif self.images_count == self.animation_time * 4 + 1:
            self.image = self.images[0]
            self.images_count = 0

        if self.tilt <= -80:
            self.image = self.images[1]
            self.images_count = self.animation_time * 2

        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)

class Pipe:
    GAP = 200
    velocity = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_image, False, True)
        self.PIPE_BOTTOM = pipe_image
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.velocity

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False

class Base:
    velocity = 5
    width = base_image.get_width()
    image = base_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.image, (self.x1, self.y))
        win.blit(self.image, (self.x2, self.y))

def draw_window(win, birds, pipes, base, score):
    win.blit(bg_image, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    text = font.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (window_width - 10 - text.get_width(), 10))
    base.draw(win)
    
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    birds = []
    nets = []
    ge = []
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)
        
    base = Base(730)
    pipes = [Pipe(700)]

    score = 0
    win = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break
                
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()
                
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
                    
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
                
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
                
            pipes.append(Pipe(700))

        for r in rem:
            pipes.remove(r)
        
        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        for pipe in pipes:
            pipe.move()
        base.move()
        draw_window(win, birds, pipes, base, score)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    winner = population.run(main, 50)

if __name__ == "__main__":
    local_directory = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_directory, "config-forward_feed.txt")
    
    print(f"Config path: {config_path}")  # Debugging line
    if not os.path.isfile(config_path):
        print("Config file not found!")
    else:
        run(config_path)
