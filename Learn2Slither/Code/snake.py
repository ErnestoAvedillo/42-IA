import pygame as pg
from motor_snake import MotorSnake

DIRECTIONS = {
    "UP": [0, -1],
    "DOWN":  [0, 1],
    "LEFT": [-1, 0],
    "RIGHT":  [1, 0]
}

ALL = 0
GREEN = 1
RED = 2

ACTIONS = ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]


class Snake(pg.sprite.Sprite, MotorSnake):
    def __init__(self, x, y, Nr_cells=[10, 10]):
        pg.sprite.Sprite.__init__(self)
        MotorSnake.__init__(self, Nr_cells)
        self.size_cells = [x // self.nr_cells[0], y // self.nr_cells[1]]
        pg.display.set_caption("Snake Game")
        pg.font.init()
        self.font = pg.font.Font(None, 36)
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((x, y))
        self.running = True

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
