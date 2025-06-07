import random
from collections import deque
import os
from directions import Directions
from collisions import Collision
import platform

ALL = 0
GREEN = 1
RED = 2


class MotorSnake():
    def __init__(self, Nr_cells=[10, 10]):
        self.nr_cells = Nr_cells
        self.direction = Directions.get_random_direction()
        self.worn = deque()
        self.red_apples = []
        self.green_apples = []
        MotorSnake.reset(self)
        self._create_map()
        self.moves = 0

    def reset(self):
        self.worn.clear()
        self._place_worn()
        self.red_apples.clear()
        self.green_apples.clear()
        self._place_apple(type=ALL)
        self.collision = Collision.NONE
        self.running = True
        self.moves = 0
        self._create_map()

    def _place_worn(self):
        self.direction = Directions.get_random_direction()
        position = self._get_rand_pos()
        position[0] = max(2, position[0])
        position[0] = min(self.nr_cells[0] - 3, position[0])
        position[1] = max(2, position[1])
        position[1] = min(self.nr_cells[1] - 3, position[1])
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
        return [random.randint(0, self.nr_cells[0] - 2),
                random.randint(0, self.nr_cells[1] - 2)]

    def _move(self):
        self.moves += 1
        new_position = [self.worn[0][0] + self.direction[0],
                        self.worn[0][1] + self.direction[1]]
        self._check_collisions(new_position)
        if (self.collision == Collision.WALL or
                self.collision == Collision.BODY):
            return self.collision, not self.running
        self.worn.appendleft(new_position)
        match self.collision:
            case Collision.GREEN_APPLE:
                self._place_apple(GREEN, new_position)
            case Collision.RED_APPLE:
                self._place_apple(RED, new_position)
                self.worn.pop()
            case Collision.NONE:
                self.worn.pop()
        return self.collision, not self.running

    def _check_collisions(self, head_pos):
        # No collision == 0
        # Collision red apple == 1
        # Collision green apple == 2
        # Collission with body = 3
        self.collision = Collision.NONE
        if head_pos in self.green_apples:
            self.collision = Collision.GREEN_APPLE
        if head_pos in self.red_apples:
            self.worn.pop()
            self.collision = Collision.RED_APPLE
            if len(self.worn) == 0:
                self.running = False
        if (head_pos[0] < 0 or
           head_pos[0] >= self.nr_cells[0] or
           head_pos[1] < 0 or
           head_pos[1] >= self.nr_cells[1]):
            self.collision = Collision.WALL
            self.running = False
        for part in self.worn:
            if head_pos == part:
                self.collision = Collision.BODY
                self.running = False

    def _create_map(self):
        self.map = [["O" for _ in range(self.nr_cells[0] + 2)]
                    for _ in range(self.nr_cells[1] + 2)]
        for i in range(self.nr_cells[0] + 2):
            self.map[0][i] = "W"
            self.map[self.nr_cells[0] + 1][i] = "W"
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
            self.map[pos[1] + 1][pos[0] + 1] = letter
            if first:
                letter = "S"

    def print_map_in_shell(self):
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system('clear')
        print("\033[H", end="")
        for fila in self.map:
            print(" ".join(f"{celda:2}" for celda in fila))

    def get_moves(self):
        return self.moves