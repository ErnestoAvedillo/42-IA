import pygame as pg
from motor_snake import MotorSnake
import numpy as np

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
        self.grass = None
        self.green_apple = None
        self.red_apple = None
        self.head_worn = None
        self.body_worn = None
        self.tail_worn = None
        self.corner_worn = None
    
    def load_grass(self, filename):
        self.grass = pg.image.load(filename)
        self.grass = pg.transform.scale(self.grass, (self.size_cells[0], self.size_cells[1]))

    def load_green_apple(self, filename):
        self.green_apple = pg.image.load(filename)
        self.green_apple = pg.transform.scale(self.green_apple, (self.size_cells[0], self.size_cells[1]))
    
    def load_red_apple(self, filename):
        self.red_apple = pg.image.load(filename)
        self.red_apple = pg.transform.scale(self.red_apple, (self.size_cells[0], self.size_cells[1]))
    
    def load_head_worn(self, filename):
        self.head_worn = pg.image.load(filename)
        self.head_worn = pg.transform.scale(self.head_worn, (self.size_cells[0], self.size_cells[1]))
    
    def orient_head_worn(self):
        if self.head_worn is not None:
            match self.direction:
                case [0, 1]:  # DOWN
                    head_worn = pg.transform.rotate(self.head_worn, 0)
                case [0, -1]:  # UP
                    head_worn = pg.transform.rotate(self.head_worn, 180)
                case [-1, 0]:  # LEFT
                    head_worn = pg.transform.rotate(self.head_worn, -90)
                case [1, 0]:  # RIGHT
                    head_worn = pg.transform.rotate(self.head_worn, 90)
            head_worn = pg.transform.scale(head_worn, (self.size_cells[0], self.size_cells[1]))        
        return head_worn
    
    def load_body_worn(self, filename):
        self.body_worn = pg.image.load(filename)
        self.body_worn = pg.transform.scale(self.body_worn, (self.size_cells[0], self.size_cells[1]))
    
    def orient_body_worn(self,index_position):
        pre_orientation = np.array(self.worn[index_position + 1]) - np.array(self.worn[index_position])
        post_orientation = np.array(self.worn[index_position - 1]) - np.array(self.worn[index_position])
        if self.body_worn is not None:
            if (np.all(pre_orientation == np.array([0, -1])) and np.all(post_orientation == np.array([0, 1])) 
                or np.all(pre_orientation == np.array([0, -1])) and np.all(post_orientation == np.array([0, -1]))
                or (np.all(pre_orientation == np.array([0, 1])) and np.all(post_orientation == np.array([0, 1])))
                or np.all(pre_orientation == np.array([0, 1])) and np.all(post_orientation == np.array([0, -1]))):
                oriented_body_worn = pg.transform.rotate(self.body_worn, 0)
            elif (np.all(pre_orientation == np.array([1,0])) and np.all(post_orientation == np.array([1, 0]))
                  or np.all(pre_orientation == np.array([1, 0])) and np.all(post_orientation == np.array([-1, 0]))
                    or np.all(pre_orientation == np.array([-1, 0])) and np.all(post_orientation == np.array([1, 0]))
                  or np.all(pre_orientation == np.array([-1, 0])) and np.all(post_orientation == np.array([-1, 0]))):
                oriented_body_worn = pg.transform.rotate(self.body_worn, 90)
                oriented_body_worn = pg.transform.scale(oriented_body_worn, (self.size_cells[0], self.size_cells[1]))
            elif (np.all(pre_orientation == np.array([0, 1])) and np.all(post_orientation == np.array([1, 0]))
                  or np.all(pre_orientation == np.array([1, 0])) and np.all(post_orientation == np.array([0, 1]))):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 0)
                oriented_body_worn = pg.transform.scale(oriented_body_worn, (self.size_cells[0], self.size_cells[1]))
            elif (np.all(pre_orientation == np.array([0, -1])) and np.all(post_orientation == np.array([-1, 0]))
                  or np.all(pre_orientation == np.array([-1, 0])) and np.all(post_orientation == np.array([0, -1]))):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 180)
                oriented_body_worn = pg.transform.scale(oriented_body_worn, (self.size_cells[0], self.size_cells[1]))
            elif (np.all(pre_orientation == np.array([0, 1])) and np.all(post_orientation == np.array([-1, 0]))
                  or np.all(pre_orientation == np.array([-1, 0])) and np.all(post_orientation == np.array([0, 1]))):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, -90)
                oriented_body_worn = pg.transform.scale(oriented_body_worn, (self.size_cells[0], self.size_cells[1]))
            elif (np.all(pre_orientation == np.array([0, -1])) and np.all(post_orientation == np.array([1, 0]))
                  or np.all(pre_orientation == np.array([1, 0])) and np.all(post_orientation == np.array([0, -1]))):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 90)
                oriented_body_worn = pg.transform.scale(oriented_body_worn, (self.size_cells[0], self.size_cells[1]))
        else:
            oriented_body_worn = self.body_worn
        return oriented_body_worn
    
    def load_tail_worn(self, filename):
        self.tail_worn = pg.image.load(filename)
        self.tail_worn = pg.transform.scale(self.tail_worn, (self.size_cells[0], self.size_cells[1]))
    
    def orient_tail_worn(self):
        len_worn = len(self.worn) - 1
        pre_orientation = np.array(self.worn[len_worn]) - np.array(self.worn[len_worn - 1])
        if self.tail_worn is not None:
            match pre_orientation.tolist():
                case [0, 1]:
                    rot_tail = pg.transform.rotate(self.tail_worn, 0)
                case [0, -1]:
                    rot_tail = pg.transform.rotate(self.tail_worn, 180)
                case [-1, 0]:
                    rot_tail = pg.transform.rotate(self.tail_worn, -90)
                case [1, 0]:
                    rot_tail = pg.transform.rotate(self.tail_worn, 90)
            rot_tail = pg.transform.scale(rot_tail, (self.size_cells[0], self.size_cells[1]))
        return rot_tail
    
    def load_corner_worn(self, filename):
        self.corner_worn = pg.image.load(filename)
        self.corner_worn = pg.transform.scale(self.corner_worn, (self.size_cells[0], self.size_cells[1]))

    def _render(self):
        for apple in self.green_apples:
            if self.green_apple is not None:
                self.screen.blit(self.green_apple, (apple[0] * self.size_cells[0],
                                 apple[1] * self.size_cells[1]))
            else:
                pg.draw.rect(self.screen, "green",
                         pg.Rect(apple[0] * self.size_cells[0],
                                 apple[1] * self.size_cells[1],
                                 self.size_cells[0],
                                 self.size_cells[1]))
        for apple in self.red_apples:
            if self.red_apple is not None:
                self.screen.blit(self.red_apple, (apple[0] * self.size_cells[0],
                                 apple[1] * self.size_cells[1]))
            else:
                pg.draw.rect(self.screen, "red",
                             pg.Rect(apple[0] * self.size_cells[0],
                                     apple[1] * self.size_cells[1],
                                     self.size_cells[0],
                                     self.size_cells[1]))
        first = True
        for part,i in zip(self.worn,range(len(self.worn))):
            if self.head_worn is not None:
                if first:
                    rot_head = self.orient_head_worn()
                    self.screen.blit(rot_head, (part[0] * self.size_cells[0],
                                                       part[1] * self.size_cells[1],
                                                self.size_cells[0],
                                                self.size_cells[1]))
                    first = False
                elif i == len(self.worn) - 1:
                    rot_tail = self.orient_tail_worn()
                    self.screen.blit(rot_tail, (part[0] * self.size_cells[0],
                                                      part[1] * self.size_cells[1],
                                                      self.size_cells[0],
                                                      self.size_cells[1]))
                else:
                    rot_body = self.orient_body_worn(i)
                    self.screen.blit(rot_body, (part[0] * self.size_cells[0],
                                                part[1] * self.size_cells[1],
                                                self.size_cells[0],
                                                self.size_cells[1]))
            else:
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
            self.screen.blit(self.grass, (0, 0))
            for i in range(self.nr_cells[0]):
                for j in range(self.nr_cells[1]):
                    self.screen.blit(self.grass, (i * self.size_cells[0],
                                                   j * self.size_cells[1]))
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
    game.load_grass("./icons/grass.png")
    game.load_green_apple("./icons/green-apple-48.png")
    game.load_red_apple("./icons/red-apple-48.png")
    game.load_head_worn("./icons/head1.png")
    game.load_body_worn("./icons/Body.png")
    game.load_tail_worn("./icons/Tail.png")
    game.load_corner_worn("./icons/corner1.png")
    game.run()
