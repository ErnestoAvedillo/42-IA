import pygame as pg
import random
from collections import deque
import os

DIRECTIONS = {
    "UP": [0, -1],
    "DOWN":  [0, 1],
    "LEFT": [-1, 0],
    "RIGHT":  [1, 0]
}

ALL = 0
GREEN = 1
RED = 2

ACTIONS =["NONE", "UP", "DOWN", "LEFT", "RIGHT"]

class Snake(pg.sprite.Sprite):
    def __init__(self, x, y, Nr_cells=[10, 10]):
        super().__init__()
        pg.display.set_caption("Snake Game")
        pg.font.init()
        self.font = pg.font.Font(None, 36)
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((x, y))
        self.running = True
        self.dt = 0
        self.nr_cells = Nr_cells
        self.size_cells = [x // self.nr_cells[0], y // self.nr_cells[1]]
        self.direction = DIRECTIONS["RIGHT"]
        self.worn = deque()
        self.red_apples = []
        self.green_apples = []
        self.reset()

    def reset(self):
        self.worn.clear()
        self._place_worn()
        self.red_apples.clear()
        self.green_apples.clear()
        self._place_apple(type=ALL)
        self.collision = 0
        self.state = 0

    def step(self, action="NONE"):
        if action in DIRECTIONS.keys():
            self.direction = DIRECTIONS[action]
        self._move()
        self._create_map()
        raw = self.map[self.worn[0]]
        col = [row[self.worn[1]] for row in self.map]
        observation = raw + col
        reward = -abs(self.state)
        return reward, 

    def _create_map(self):
        self.map = [["O" for _ in range(self.nr_cells[0] + 2)]
                    for _ in range(self.nr_cells[1] + 2)]
        for i in range(self.nr_cells[0] + 2):
            self.map[0][i] = "W"
            self.map[self.nr_cells[0]][i] = "W"
        for i in range(self.nr_cells[1] + 2):
            self.map[i][0] = "W"
            self.map[i][self.nr_cells[0] + 1] = "W"
        for pos in self.red_apples:
            self.map[pos[1] + 1][pos[0] + 1] = "R"
        for pos in self.green_apples:
            self.map[pos[1] + 1][pos[0] + 1] = "G"
        letter = "H"
        first = True
        for pos in self.worn:
            self.map[pos[1] + 1] [pos[0] + 1]= letter
            if first:
                letter = "S"

    def _place_worn(self):
        self.direction = DIRECTIONS[random.choice(list(DIRECTIONS.keys()))]
        position = self._get_rand_pos()
        position[0] = max(2, position[0])
        position[0] = min(self.nr_cells[0] - 2, position[0])
        position[1] = max(2, position[1])
        position[1] = min(self.nr_cells[1] - 2, position[1])
        self.worn.append(position)
        self.worn.append([position[0] - self.direction[0],
                          position[1] - self.direction[1]])
        self.worn.append([position[0] - self.direction[0] * 2,
                          position[1] - self.direction[1] * 2])

    def _place_apple(self, type, pos=None, operation="replace"):
        if type == ALL:
            self.red_apples.append(self._get_rand_pos())
            for i in range(2):
                posicion = self._get_rand_pos()
                while (posicion in self.red_apples or
                       posicion in self.green_apples or
                       posicion in self.worn):
                    posicion = self._get_rand_pos()
                self.green_apples.append(posicion)
        elif type == GREEN:
            if operation == "replace":
                for p in self.green_apples:
                    if p == pos:
                        self.green_apples.remove(p)
            posicion = self._get_rand_pos()
            while (posicion in self.red_apples or
                   posicion in self.green_apples or
                   posicion in self.worn):
                posicion = self._get_rand_pos()
            self.green_apples.append(posicion)
        elif type == RED:
            if operation == "replace":
                for p in self.red_apples:
                    if p == pos:
                        self.red_apples.remove(p)
            posicion = self._get_rand_pos()
            while (posicion in self.red_apples or
                   posicion in self.green_apples or
                   posicion in self.worn):
                posicion = self._get_rand_pos()
            self.red_apples.append(posicion)

    def _get_rand_pos(self):
        return [random.randint(0, self.nr_cells[0] - 1),
                random.randint(0, self.nr_cells[1] - 1)]

    def _move(self):
        new_position = [self.worn[0][0] + self.direction[0],
                        self.worn[0][1] + self.direction[1]]
        self._check_collisions(new_position)
        if self.collision == 3:
            return
        self.worn.appendleft(new_position)
        match self.collision:
            case 1:
                self._place_apple(GREEN, new_position)
            case 2:
                self._place_apple(RED, new_position)
                self.worn.pop()
            case _:
                self.worn.pop()

    def _render(self):
        for apple in self.green_apples:
            pg.draw.rect(self.screen, "green",
                         pg.Rect(apple[0] * self.size_cells[0],
                                 apple[1] * self.size_cells[1],
                                 self.size_cells[0],
                                 self.size_cells[1]))
        for apple in self.red_apples:
            pg.draw.rect(self.screen, "red",
                         pg.Rect(apple[0] * self.size_cells[0],
                                 apple[1] * self.size_cells[1],
                                 self.size_cells[0],
                                 self.size_cells[1]))
        first = True
        for part in self.worn:
            if first:
                color = "gray"
                first = False
            else:
                color = "blue"
            pg.draw.rect(self.screen, color,
                         pg.Rect(part[0] * self.size_cells[0],
                                 part[1] * self.size_cells[1],
                                 self.size_cells[0],
                                 self.size_cells[1]))

    def print_map_in_shell(self):
        os.system('clear')
        print("\033[H", end="")
        for fila in self.map:
            print(" ".join(f"{celda:2}" for celda in fila))

    def _check_collisions(self, head_pos):
        # No collision == 0
        # Collision red apple == 1
        # Collision green apple == 2
        # Collission with body = 3
        self.collision = 0
        if head_pos in self.green_apples:
            self.collision = 1
        if head_pos in self.red_apples:
            self.worn.pop()
            self.collision = 2
            if len(self.worn) == 0:
                self.running = False
        if (head_pos[0] < 0 or
           head_pos[0] >= self.nr_cells[0] or
           head_pos[1] < 0 or
           head_pos[1] >= self.nr_cells[1]):
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
            self._create_map()
            self.print_map_in_shell()
            pg.display.flip()
            self.clock.tick(2)
        font = pg.font.SysFont('Arial', 48)
        text_surface = font.render("Game Over", True, (255, 255, 255))
        self.screen.blit(text_surface,
                         ((self.nr_cells[0] // 2 - 4) * self.size_cells[0],
                          (self.nr_cells[1] // 2 - 1) * self.size_cells[1]))
        pg.display.flip()  # Update the full display
        self.clock.tick(1)
        pg.quit()


if __name__ == "__main__":
    game = Snake(800, 600, [10, 10])
    game.run()
