
import tkinter as tk
import random

# Constants
GRID_SIZE = 10
CELL_SIZE = 40
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
MOVE_INTERVAL = 200  # milliseconds

# Directions: (dy, dx)
DIRECTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1)
}


class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.reset_game()
        self.root.bind("<KeyPress>", self.change_direction)
        self.running = True
        self.game_loop()

    def reset_game(self):
        self.snake = [(5, 5)]
        self.direction = "Right"
        self.food = self.place_food()
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        # Draw snake
        for y, x in self.snake:
            self.draw_cell(x, y, "green")
        # Draw food
        fy, fx = self.food
        self.draw_cell(fx, fy, "red")

    def draw_cell(self, x, y, color):
        x1 = x * CELL_SIZE
        y1 = y * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def place_food(self):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1),
                   random.randint(0, GRID_SIZE - 1))
            if pos not in self.snake:
                return pos

    def change_direction(self, event):
        new_dir = event.keysym
        opposites = {"Up": "Down",
                     "Down": "Up",
                     "Left": "Right",
                     "Right": "Left"}
        if new_dir in DIRECTIONS and new_dir != opposites.get(self.direction):
            self.direction = new_dir

    def move_snake(self):
        dy, dx = DIRECTIONS[self.direction]
        head_y, head_x = self.snake[-1]
        new_head = (head_y + dy, head_x + dx)

        # Check collision with walls or self
        if (
            not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE)
            or new_head in self.snake
        ):
            self.game_over()
            return

        self.snake.append(new_head)

        if new_head == self.food:
            self.food = self.place_food()
        else:
            self.snake.pop(0)

        self.draw_board()

    def game_over(self):
        self.running = False
        self.canvas.create_text(WIDTH//2,
                                HEIGHT//2,
                                text="GAME OVER",
                                fill="white",
                                font=("Arial", 24))

    def game_loop(self):
        if self.running:
            self.move_snake()
            self.root.after(MOVE_INTERVAL, self.game_loop)


# Run the game
root = tk.Tk()
root.title("Snake Game - 10x10")
game = SnakeGame(root)
root.mainloop()
