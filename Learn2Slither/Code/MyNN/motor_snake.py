import random
from collections import deque
import os
from directions import Directions
from collisions import Collision
import platform

ALL = 0
GREEN = 1
RED = 2
MAX_MOVES = 1000


class MotorSnake():
    def __init__(self, Nr_cells=[10, 10]):
        self.nr_cells = Nr_cells
        self.direction = Directions.get_random_direction()
        self.worn = deque()
        self.history = deque(maxlen=20)  # Store last 20 worn deques
        self.red_apples = []
        self.green_apples = []
        self.reset(self)

    def reset(self):
        """Reset the game state."""
        self.worn.clear()
        self._place_worn()
        self.red_apples.clear()
        self.green_apples.clear()
        self._place_apple(type=ALL, operation="not_replace")
        self.collision = Collision.NONE
        self.termnate = False
        self.moves = 0
        self.red_apples_eaten = 0
        self.green_apples_eaten = 0
        self._create_map()

    def _place_worn(self):
        """Place the worn in a random position."""
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
        """Place an apple in a random position.
        pos is the appel to be replaced. only is used if operation is
            "replace".
        If operation is "replace", the apple will be replaced in an
            aleatory position.
        If operation is "not_replace", the apple will not be added in an
            aleatory position.
        Args:
            type (int): Type of apple to place (GREEN, RED, ALL).
            pos (list, optional): Position to place the apple.
            Defaults to None.
            Operation (str, optional): Operation to perform with the apple.
        """
        if type == GREEN or type == ALL:

            if operation == "replace":
                for p in self.green_apples:
                    if p == pos:
                        self.green_apples.remove(p)
            if type == ALL:
                nr_apples = 2
            else:
                nr_apples = 1
            for i in range(nr_apples):
                posicion = self._get_rand_pos()
                while (posicion in self.red_apples or
                       posicion in self.green_apples or
                       posicion in self.worn):
                    posicion = self._get_rand_pos()
                self.green_apples.append(posicion)
        if type == RED or type == ALL:
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
        """Get a random position in the map."""
        return [random.randint(0, self.nr_cells[0] - 2),
                random.randint(0, self.nr_cells[1] - 2)]

    def _move(self):
        """Move the snake in the current direction.
        Returns:
            Collision: The type of collision that occurred.
            bool: True if the game is terminated, False otherwise.
        """
        self.moves += 1
        if len(self.worn) == 0:
            self.termnate = True
            return Collision.NONE, self.termnate
        new_position = [self.worn[0][0] + self.direction[0],
                        self.worn[0][1] + self.direction[1]]
        self._check_collisions(new_position)
        if (self.collision == Collision.WALL or
                self.collision == Collision.BODY):
            self.termnate = True
            return self.collision, self.termnate
        self.worn.appendleft(new_position)
        match self.collision:
            case Collision.GREEN_APPLE:
                self._place_apple(GREEN, new_position)
                return self.collision, self.termnate
            case Collision.RED_APPLE:
                self.worn.pop()
                self.worn.pop()
                if len(self.worn) == 0:
                    self.termnate = True
                    return self.collision, self.termnate
                self._place_apple(RED, new_position)
            case _:
                self.worn.pop()
        self.history.append(self.worn.copy())
        return self.collision, self.termnate

    def check_head_psition_near_green_apple(self):
        """ Check if the head position is near a green apple.
        This means that the head is in the same row or column as a green apple,
        and the distance to the green apple is less than 2 cells.
        """
        if len(self.worn) == 0 or len(self.green_apples) == 0:
            return False
        for i in range(len(self.green_apples)):
            dx = abs(self.worn[0][0] - self.green_apples[i][0])
            dy = abs(self.worn[0][1] - self.green_apples[i][1])
            if ((self.worn[0][0] == self.green_apples[i][0] and dy == 1) or
                    (self.worn[0][1] == self.green_apples[i][1] and dx == 1)):
                return True
        return False

    def check_head_psition_alligned_with_green_apple(self):
        """ Check if the head position is alligned with a green apple.
        This means that the head is in the same row or column as a green apple.
        """
        if len(self.worn) == 0 or len(self.green_apples) == 0:
            return False
        for i in range(len(self.green_apples)):
            if ((self.worn[0][0] == self.green_apples[i][0] or
                 self.worn[0][1] == self.green_apples[i][1])):
                return True
        return False

    def _check_collisions(self, head_pos):
        """Check for collisions with walls, apples, or the snake's body.
        # No collision == 0
        # Collision red apple == 1
        # Collision green apple == 2
        # Collission with body = 3
        # Collision with wall = 4
        Args:
            head_pos (list): The position of the snake's head.
        Returns:
            None: If no collision occurs.
            Collision: The type of collision that occurred.
            bool: True if the game is terminated, False otherwise.
        """
        if self.moves >= MAX_MOVES or self.moves >= len(self.worn) * 100:
            self.termnate = True
            self.collision = Collision.BODY
            return
        self.collision = Collision.NONE
        if (head_pos[0] < 0 or
           head_pos[0] >= self.nr_cells[0] or
           head_pos[1] < 0 or
           head_pos[1] >= self.nr_cells[1]):
            self.collision = Collision.WALL
            self.termnate = True
            return
        for part in self.worn:
            if head_pos == part:
                self.collision = Collision.BODY
                self.termnate = True
                return
        if head_pos in self.red_apples:
            self.collision = Collision.RED_APPLE
            self.red_apples_eaten += 1
            if len(self.worn) == 0:
                self.termnate = True
                return
            return
        if head_pos in self.green_apples:
            self.collision = Collision.GREEN_APPLE
            self.green_apples_eaten += 1
            return
        if self.check_head_psition_near_green_apple():
            self.collision = Collision.IS_THE_WAY
            return
        if self.check_head_psition_alligned_with_green_apple():
            self.collision = Collision.IS_ALLIGNED_WITH_GREEN_APPLE
            return
        if self.worn_has_repeated_position():
            self.collision = Collision.REPEATED_POSITION
        return

    def _create_map(self):
        """Create the map of the game.
        The map is a 2D list with the following structure:
        - "W" for walls
        - "O" for empty cells
        - "R" for red apples
        - "G" for green apples
        - "H" for the head of the snake
        - "S" for the rest of the snake
        """
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
        return

    def print_map_in_shell(self, clear=True):
        """Print the map in the shell.
        Args:
            clear (bool): If True, clear the shell before printing.
        """
        if clear:
            if platform.system() == "Windows":
                os.system("cls")
            else:
                os.system('clear')
        print("\033[H", end="")
        for fila in self.map:
            print(" ".join(f"{celda:2}" for celda in fila))

    def get_moves(self):
        """Get the number of moves made in the game.
        Returns:
            int: The number of moves made.
        """
        return self.moves

    def get_length_worn(self):
        """Get the length of the worn.
        Returns:
            int: The length of the worn.
        """
        return len(self.worn)

    def worn_has_repeated_position(self):
        """
        Check if the worn has repeated positions.
        Returns:
            bool: True if there are repeated positions, False otherwise.
        """
        nr_repeated_positions = 0
        for historic_worn in self.history:
            if historic_worn == self.worn:
                nr_repeated_positions += 1
        if nr_repeated_positions > 2:
            return True
        return False

    def get_statistics(self):
        """ Returns the statistics of the game
        Returns:
            dict: A dictionary with the statistics of the game.
            - "score": The score of the game (length of the worn).
            - "moves": The number of moves made in the game.
            - "green_apples": The number of green apples eaten.
            - "red_apples": The number of red apples eaten.
        """
        info = {
            "score": len(self.worn),
            "moves": self.moves,
            "green_apples": self.green_apples_eaten,
            "red_apples": self.red_apples_eaten,
        }
        return info
