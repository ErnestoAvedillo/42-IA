import pygame as pg
import random
from collections import deque

GREEN = 1
RED = 2
BOTH = 3

DIRECTIONS = {
    "UP": [0,-1],
    "DOWN":  [0,1],
    "LEFT": [-1,0],
    "RIGHT":  [1,0]
}

class Snake(pg.sprite.Sprite):
    def __init__(self, x, y, Nr_cells=[10,10]):
        super().__init__()
        pg.display.set_caption("Snake Game")
        pg.font.init()
        self.font = pg.font.Font(None, 36)
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((x, y))
        self.running = True
        self.dt = 0
        self.nr_cells = Nr_cells
        self.size_cells = (x // self.nr_cells[0], y // self.nr_cells[1])
        self.direction = DIRECTIONS["RIGHT"]
        self.map = [[0 for _ in range(Nr_cells[0])] for _ in range(Nr_cells[1])]
        self.green_apple_pos = [random.randint(0, self.nr_cells[0] - 1),random.randint(0, self.nr_cells[1] - 1)]
        self.red_apple_pos = [random.randint(0, self.nr_cells[0] - 1),random.randint(0, self.nr_cells[1] - 1)]
        self.worn = deque()
        self.worn.append([Nr_cells[0] // 2, Nr_cells[1] // 2])
        self.worn.append([Nr_cells[0] // 2 - 1, Nr_cells[1] // 2])
        self._place_apple(BOTH)
        self.collision = 0

    def _place_apple(self,type):
        if type == RED:
            print(f"Placing red apple currently at {self.red_apple_pos}")
            self.red_apple_pos = self._get_rand_pos()
            print(f"Placing red apple at {self.red_apple_pos}")
            while (self.green_apple_pos == self.red_apple_pos) :
                print(f"Repeated placing red apple collision with green apple at {self.red_apple_pos}")
                self.red_apple_pos = self._get_rand_pos()
                print(f"Repeated placing red apple collision with green apple at {self.red_apple_pos}")
        elif type == GREEN:
            print(f"Placing green apple currently at {self.green_apple_pos}")
            self.green_apple_pos = self._get_rand_pos()
            print(f"Placing green apple at {self.green_apple_pos}")
            while (self.green_apple_pos == self.red_apple_pos):
                print(f"Repeated placing green apple collision with red apple at {self.green_apple_pos}")
                self.geen_apple_pos = self._get_rand_pos()
                print(f"Repeated placing green apple collision with red apple at {self.green_apple_pos}")
        else:
            self.red_apple_pos = self._get_rand_pos()
            self.green_apple_pos = self._get_rand_pos()
            while (self.green_apple_pos == self.red_apple_pos):
                self.red_apple_pos = self._get_rand_pos()
        for part in self.worn:
            if (self.green_apple_pos == part):
                self.green_apple_pos = self._place_apple(GREEN)
            if (self.red_apple_pos == part):
                self.red_apple_pos = self._place_apple(RED)

    def _get_rand_pos (self):
        return [random.randint(0, self.nr_cells[0] - 1),random.randint(0, self.nr_cells[1] - 1)]

    def _move(self):
        new_position =  [self.worn[0][0] + self.direction[0], self.worn[0][1] + self.direction[1]]
        self._check_collisions(new_position)
        if self.collision == 3:
            return
        self.worn.appendleft(new_position)
        match self.collision:
            case 1:
                self._place_apple(GREEN)
            case 2:
                self._place_apple(RED)
                self.worn.pop()
            case _:
                self.worn.pop()

    def _render(self):
        pg.draw.rect(self.screen, "green", pg.Rect(self.green_apple_pos[0] * self.size_cells[0], 
                                                   self.green_apple_pos[1] * self.size_cells[1], 
                                                   self.size_cells[0], 
                                                   self.size_cells[1]))
        pg.draw.rect(self.screen, "red", pg.Rect(self.red_apple_pos[0] * self.size_cells[0], 
                                                   self.red_apple_pos[1] * self.size_cells[1], 
                                                   self.size_cells[0], 
                                                   self.size_cells[1]))
        for part in self.worn:
            pg.draw.rect(self.screen, "blue", pg.Rect(part[0] * self.size_cells[0], 
                                                       part[1] * self.size_cells[1], 
                                                       self.size_cells[0], 
                                                       self.size_cells[1]))
    def _check_collisions(self, head_pos):
        # No collision == 0
        # Collision red apple == 1
        # Collision green apple == 2
        # Collission with body = 3
        self.collision = 0
        if head_pos == self.green_apple_pos:
            self.collision = 1
        elif head_pos == self.red_apple_pos:
            self.worn.pop()
            self.collision = 2
        if head_pos[0] < 0 or head_pos[0] >= self.nr_cells[0] or head_pos[1] < 0 or head_pos[1] >= self.nr_cells[1]:
            self.collision = 3
            self.running = False
        for part in self.worn:
            if head_pos == part:
                self.collision = 3
                self.running = False

    def _check_event_(self):
        events = pg.event.get()
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    self.direction = DIRECTIONS["LEFT"]
                if event.key == pg.K_RIGHT:
                    self.direction = DIRECTIONS["RIGHT"]
                if event.key == pg.K_UP:
                    self.direction = DIRECTIONS["UP"]
                if event.key == pg.K_DOWN:
                    self.direction = DIRECTIONS["DOWN"]
                if event.key == pg.K_ESCAPE:
                    self.running = False


    def run(self):
        while self.running:
            self._check_event_()
            self._move()
            self.screen.fill("purple")
            self._render()
            pg.display.flip()
            self.clock.tick(5)
        pg.quit()

if __name__ == "__main__":
    game = Snake(800,600,[30,30])
    game.run()