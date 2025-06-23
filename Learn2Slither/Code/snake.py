import pygame as pg
from motor_snake import MotorSnake
import numpy as np
from dqn_agent_keras import DQNAgent
from actions import Action
from directions import Directions
from rewards import Reward
import time
import os
import argparse

MAX_MOVES = 1000  # Maximum number of moves before truncation

DICTIONARY_OBSERVATION = {
    "O": 1,  # Empty cell
    "W": 2,  # Wall
    "H": 3,  # Head of the snake
    "S": 4,  # Body of the snake
    "G": 5,  # Green apple
    "R": 6   # Red apple
}

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
    def __init__(self, x, y, Nr_cells=[10, 10], modelname=None):
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
        self.menu_active = True
        self.truncated = False
        self._autoplaying = False
        self.agent = DQNAgent(state_shape=len(self.get_observation()),
                 num_actions=Action.get_len_actions())
        if modelname is None:
            print(f"Model for auto-play not added. No autoplay will be performed.")
            self.show_autoplay = False
        else:
            self.show_autoplay = True
            self.agent.load_model(modelname)
    
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
            if event.type == pg.QUIT:
                pg.quit()
                exit()
            if self.menu_active:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_1:
                        self.run()
                        return
                    elif event.key == pg.K_2:
                        self._auto_play()
                        return
                    elif event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if self.manual_button.collidepoint(pg.mouse.get_pos()):
                        self.run()
                        return
                    elif self.auto_button.collidepoint(pg.mouse.get_pos()):
                        self._auto_play()
                        return
                pg.event.clear()
            if self._autoplaying:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self._autoplaying = False
                        self.episode_over = True
                        return
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

    def _print_grass(self):
        for i in range(self.nr_cells[0]):
            for j in range(self.nr_cells[1]):
                self.screen.blit(self.grass, 
                                 (i * self.size_cells[0],
                                  j * self.size_cells[1]))
    
    def _print_gameover(self):
        font = pg.font.SysFont('Arial', 48)
        text_surface = font.render("Game Over", True, (255, 255, 255))
        self.screen.blit(text_surface,
                         ((self.nr_cells[0] // 2 - 4) * self.size_cells[0],
                          (self.nr_cells[1] // 2 - 1) * self.size_cells[1]))
        pg.display.flip()  # Update the full display
        self.clock.tick(1)
        self._print_grass()
        self.reset()
        self._render()
                
    def run(self):
        """Run the game in manual mode."""
        self.menu_active = False
        self.clock.tick(2)
        while self.running:
            self._check_event_()
            self._move()
            self._print_grass()
            self._render()
            pg.display.flip()
            self.clock.tick(2)
        self._print_gameover()
        self.menu_active = True

    def _auto_play(self):
        """Auto-play the game using the agent's policy."""
        self.menu_active = False
        self._autoplaying = True
        self.clock.tick(2)
        self.episode_over = False
        observation = self.get_observation()
        while not self.episode_over:
            self._check_event_()
            action, is_aleatory = self.agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = self.step(action)
            self._print_grass()
            self._render()
            pg.display.flip()
            # self.print_map_in_shell()
            self.clock.tick(2)
            if not self.episode_over:
                self.episode_over = terminated or truncated
            #time.sleep(1)
        self._autoplaying = False
        self.menu_active = True
        self._print_gameover()

    def step(self, action):
        """Perform a step in the game with the given action.
        Args:
            action (int): The action to perform, represented as an integer.
        Returns:
            tuple: A tuple containing the observation, reward, termination status,
                   truncation status, and additional info.
        """
        # convert from np.int64 to int
        action = int(action)
        self.direction = Directions.get_direction(action)
        collision, self.terminated = self._move()
        self._create_map()
        self.reward = Reward.get_reward(self, collision)
        if self.get_moves() >= MAX_MOVES:
            self.truncated = True
        return (self.get_observation(), self.reward,
                self.terminated, self.truncated, {})

    def get_observation(self):
        """
        Returns the current observation of the environment.
        The observation consists of the environment seen by the snake,
        Structure:
        head_collition_left:true/false
        head_collition_right:true/false
        head_collition_up:true/false
        head_collition_down:true/false
        green_apple_left:true/false
        green_apple_right:true/false
        green_apple_up:true/false
        green_apple_down:true/false
        red_apple_left:true/false
        red_apple_right:true/false
        red_apple_up:true/false
        red_apple_down:true/false
        """
        if len(self.worn) == 0:
            head_col = 1
            head_raw = 1
        else:
            head_col = self.worn[0][0] + 1
            head_raw = self.worn[0][1] + 1
#        raw = self.map[head_raw]
#        col = []
#        for i in range(self.nr_cells[1] + 2):
#            col.append(self.map[i][head_col])
#        numbered_raw = [DICTIONARY_OBSERVATION[cell] for cell in raw]
#        numbered_col = [DICTIONARY_OBSERVATION[cell] for cell in col]
#        observation = numbered_raw + numbered_col
        observation = [0 for _ in range(12)]
        # Check for collisions with walls
        if self.map[head_raw - 1][head_col] == "W" or \
           self.map[head_raw - 1][head_col] in "S":
            observation[0] = 1
        if self.map[head_raw + 1][head_col] == "W" or \
           self.map[head_raw + 1][head_col] in "S":
            observation[1] = 1
        if self.map[head_raw][head_col - 1] == "W" or \
           self.map[head_raw][head_col - 1] in "S":
            observation[2] = 1
        if self.map[head_raw][head_col + 1] == "W" or \
           self.map[head_raw][head_col + 1] in "S":
            observation[3] = 1
        # Check for coincidences in raw or col with green apples
        for apple in self.green_apples:
            if apple[0] < head_col and apple[1] == head_raw:
                observation[4] = 1
            if apple[0] > head_col and apple[1] == head_raw:
                observation[5] = 1
            if apple[0] == head_col and apple[1] > head_raw:
                observation[6] = 1
            if apple[0] == head_col and apple[1] < head_raw:
                observation[7] = 1
        # Check for coincidences in raw or col with red apples
        for apple in self.red_apples:
            if apple[0] < head_col and apple[1] == head_raw:
                observation[8] = 1
            if apple[0] > head_col and apple[1] == head_raw:
                observation[9] = 1
            if apple[0] == head_col and apple[1] > head_raw:
                observation[10] = 1
            if apple[0] == head_col and apple[1] < head_raw:
                observation[11] = 1
        return observation

    def _select_mode(self):
        """Display the mode selection menu."""
        self._print_grass()
        self._render()
        self.menu_font = pg.font.Font(None, 48)
        self.info_font = pg.font.Font(None, 28)

        # Define button rectangles
        self.manual_button = pg.Rect(self.screen.get_width() // 2 - 100, 180, 200, 50)
        if self.show_autoplay:
            self.auto_button = pg.Rect(self.screen.get_width() // 2 - 100, 250, 200, 50)

        while self.menu_active:
            # Title
            title = self.menu_font.render("Select Mode", True, (255, 255, 255))
            self.screen.blit(title, (self.screen.get_width() // 2 - title.get_width() // 2, 100))

            # Draw buttons
            pg.draw.rect(self.screen, (70, 130, 180), self.manual_button)  # Steel Blue
            if self.show_autoplay:
                pg.draw.rect(self.screen, (34, 139, 34), self.auto_button)     # Forest Green

            # Button text
            manual_text = self.info_font.render("Manual Mode", True, (255, 255, 255))
            auto_text = self.info_font.render("Auto Mode", True, (255, 255, 255))
            self.screen.blit(manual_text, (self.manual_button.x + 30, self.manual_button.y + 10))
            if self.show_autoplay:
                self.screen.blit(auto_text, (self.auto_button.x + 40, self.auto_button.y + 10))

            pg.display.flip()
            self._check_event_()
        pg.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for the Snake game.')
    parser.add_argument('-f', '--file_model', type=str, help='File name to save the model.')
    args = parser.parse_args()
    if not args.file_model:
        print("No model name provided.")
        modelname = None
    else:
        modelname = args.file_model
    if  not os.path.exists(modelname):
        print(f"Model file {modelname} does not exist. Starting with a new model.")
        modelname = None

    game = Snake(800, 600, [10, 10], modelname=modelname)
    game.load_grass("./icons/grass.png")
    game.load_green_apple("./icons/green-apple-48.png")
    game.load_red_apple("./icons/red-apple-48.png")
    game.load_head_worn("./icons/head1.png")
    game.load_body_worn("./icons/Body.png")
    game.load_tail_worn("./icons/Tail.png")
    game.load_corner_worn("./icons/corner1.png")

    game._select_mode()

