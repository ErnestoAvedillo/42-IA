import pygame as pg
from env_snake import EnvSnake
import numpy as np
from dqn_agent_keras import DQNAgent
from actions import Action
from directions import Directions
from rewards import Reward
import time
import os
import platform
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


MAX_MOVES = 1000  # Maximum number of moves before truncation

DICTIONARY_OBSERVATION = {
    "O": 1,  # Empty cell
    "W": 2,  # Wall
    "H": 3,  # Head of the snake
    "S": 4,  # Body of the snake
    "G": 5,  # Green apple
    "R": 6   # Red apple
}

UP = np.array([0, -1])
DOWN = np.array([0, 1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])

ALL = 0
GREEN = 1
RED = 2

ACTIONS = ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]

MODEL_DEFAULT_NAME = "Model_Q.pt"


class Snake(pg.sprite.Sprite, EnvSnake):
    def __init__(self, x, y,
                 Nr_cells=[10, 10],
                 modelname=None,
                 stats_man=None,
                 stats_auto=None,
                 stats_learn=None):
        pg.sprite.Sprite.__init__(self)
        EnvSnake.__init__(self, Nr_cells)
        self.size_cells = [x // self.nr_cells[0], y // self.nr_cells[1]]
        pg.display.set_caption("Snake Game")
        pg.font.init()
        self.font = pg.font.Font(None, 36)
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((x, y))
        self.grass = None
        self.green_apple = None
        self.red_apple = None
        self.head_worn = None
        self.body_worn = None
        self.tail_worn = None
        self.corner_worn = None
        self.menu_active = True
        self.truncated = False
        self._learning_game = False
        self._autoplaying = False
        self.stats_manual = stats_man
        self.stats_auto = stats_auto
        self.stats_learn = stats_learn
        self.agent = DQNAgent(state_shape=len(self.get_observation()),
                              num_actions=Action.get_len_actions(),
                              epsilon=0.01)
        if modelname is None or not os.path.exists(modelname):
            print("Model for auto-play not added or invalid.")
            print("No autoplay will be available.")
            self.show_autoplay = False
            self.modelname = MODEL_DEFAULT_NAME
            self.agent.set_model_name(self.modelname)
        elif os.path.exists(modelname):
            print(f"Model {modelname} loaded for auto-play.")
            self.modelname = modelname
            self.agent.load_model(self.modelname)
            self.show_autoplay = True

    def load_grass(self, filename):
        self.grass = pg.image.load(filename)
        self.grass = pg.transform.scale(self.grass,
                                        (self.size_cells[0],
                                         self.size_cells[1])
                                        )

    def load_green_apple(self, filename):
        self.green_apple = pg.image.load(filename)
        self.green_apple = pg.transform.scale(self.green_apple,
                                              (self.size_cells[0],
                                               self.size_cells[1])
                                              )

    def load_red_apple(self, filename):
        self.red_apple = pg.image.load(filename)
        self.red_apple = pg.transform.scale(self.red_apple,
                                            (self.size_cells[0],
                                             self.size_cells[1])
                                            )

    def load_head_worn(self, filename):
        self.head_worn = pg.image.load(filename)
        self.head_worn = pg.transform.scale(self.head_worn,
                                            (self.size_cells[0],
                                             self.size_cells[1]))

    def orient_head_worn(self):
        if self.head_worn is not None:
            if all(self.direction == DOWN):
                head_worn = pg.transform.rotate(self.head_worn, 0)
            elif all(self.direction == UP):
                head_worn = pg.transform.rotate(self.head_worn, 180)
            elif all(self.direction == LEFT):
                head_worn = pg.transform.rotate(self.head_worn, -90)
            elif all(self.direction == RIGHT):
                head_worn = pg.transform.rotate(self.head_worn, 90)
            head_worn = pg.transform.scale(head_worn,
                                           (self.size_cells[0],
                                            self.size_cells[1]))
        return head_worn

    def load_body_worn(self, filename):
        self.body_worn = pg.image.load(filename)
        self.body_worn = pg.transform.scale(self.body_worn,
                                            (self.size_cells[0],
                                             self.size_cells[1]))

    def orient_body_worn(self, index_position):
        pre_orientation = (np.array(self.worn[index_position + 1]) -
                           np.array(self.worn[index_position]))
        post_orientation = (np.array(self.worn[index_position - 1]) -
                            np.array(self.worn[index_position]))
        if self.body_worn is not None:
            if ((np.all(pre_orientation == UP) or
                 np.all(pre_orientation == DOWN))
                and
                (np.all(post_orientation == UP) or
                 np.all(post_orientation == DOWN))):
                oriented_body_worn = pg.transform.rotate(self.body_worn, 0)
            elif ((np.all(pre_orientation == RIGHT) or
                   np.all(pre_orientation == LEFT))
                  and
                  (np.all(post_orientation == RIGHT) or
                   np.all(post_orientation == LEFT))):
                oriented_body_worn = pg.transform.rotate(self.body_worn, 90)
            elif (np.all(pre_orientation == DOWN) and
                  np.all(post_orientation == RIGHT)
                  or
                  np.all(pre_orientation == RIGHT) and
                  np.all(post_orientation == DOWN)):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 0)
            elif (np.all(pre_orientation == UP) and
                  np.all(post_orientation == LEFT)
                  or
                  np.all(pre_orientation == LEFT) and
                  np.all(post_orientation == UP)):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 180)
            elif (np.all(pre_orientation == DOWN) and
                  np.all(post_orientation == LEFT)
                  or
                  np.all(pre_orientation == LEFT) and
                  np.all(post_orientation == DOWN)):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, -90)
            elif (np.all(pre_orientation == UP) and
                  np.all(post_orientation == RIGHT)
                  or
                  np.all(pre_orientation == RIGHT) and
                  np.all(post_orientation == UP)):
                oriented_body_worn = pg.transform.rotate(self.corner_worn, 90)
            oriented_body_worn = pg.transform.scale(oriented_body_worn,
                                                    (self.size_cells[0],
                                                     self.size_cells[1]))
        else:
            oriented_body_worn = self.body_worn
        return oriented_body_worn

    def load_tail_worn(self, filename):
        self.tail_worn = pg.image.load(filename)
        self.tail_worn = pg.transform.scale(self.tail_worn,
                                            (self.size_cells[0],
                                             self.size_cells[1])
                                            )

    def orient_tail_worn(self):
        len_worn = len(self.worn) - 1
        pre_orientation = (np.array(self.worn[len_worn]) -
                           np.array(self.worn[len_worn - 1]))
        if self.tail_worn is not None:
            if np.all(pre_orientation == DOWN):
                rot_tail = pg.transform.rotate(self.tail_worn, 0)
            elif np.all(pre_orientation == UP):
                rot_tail = pg.transform.rotate(self.tail_worn, 180)
            elif np.all(pre_orientation == LEFT):
                rot_tail = pg.transform.rotate(self.tail_worn, -90)
            elif np.all(pre_orientation == RIGHT):
                rot_tail = pg.transform.rotate(self.tail_worn, 90)
            rot_tail = pg.transform.scale(rot_tail,
                                          (self.size_cells[0],
                                           self.size_cells[1]))
        return rot_tail

    def load_corner_worn(self, filename):
        self.corner_worn = pg.image.load(filename)
        self.corner_worn = pg.transform.scale(self.corner_worn,
                                              (self.size_cells[0],
                                               self.size_cells[1]))

    def _render(self):
        for apple in self.green_apples:
            if self.green_apple is not None:
                self.screen.blit(self.green_apple,
                                 (apple[0] * self.size_cells[0],
                                  apple[1] * self.size_cells[1]))
            else:
                pg.draw.rect(self.screen, "green",
                             pg.Rect(apple[0] * self.size_cells[0],
                                     apple[1] * self.size_cells[1],
                                     self.size_cells[0],
                                     self.size_cells[1]))
        for apple in self.red_apples:
            if self.red_apple is not None:
                self.screen.blit(self.red_apple,
                                 (apple[0] * self.size_cells[0],
                                  apple[1] * self.size_cells[1]))
            else:
                pg.draw.rect(self.screen, "red",
                             pg.Rect(apple[0] * self.size_cells[0],
                                     apple[1] * self.size_cells[1],
                                     self.size_cells[0],
                                     self.size_cells[1]))
        first = True
        for part, i in zip(self.worn, range(len(self.worn))):
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
                    if event.key == pg.K_F1:
                        self.run()
                        return
                    elif event.key == pg.K_F2:
                        self._auto_play()
                        return
                    elif event.key == pg.K_F3:
                        self._learn_game()
                        return
                    elif event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if self.manual_button.collidepoint(pg.mouse.get_pos()):
                        self.run()
                        return
                    elif (self.show_autoplay and
                          self.auto_button.collidepoint(pg.mouse.get_pos())):
                        self._auto_play()
                        return
                    elif self.learn_button.collidepoint(pg.mouse.get_pos()):
                        self._learn_game()
                        return
                    elif self.stats_button.collidepoint(pg.mouse.get_pos()):
                        self._select_statistics()
                        return
                pg.event.clear()
            if self._autoplaying | self._learning_game:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self._autoplaying = False
                        self._learning_game = False
                        self.episode_over = True
                        return
                    else:
                        pass
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    self.direction = LEFT
                if event.key == pg.K_RIGHT:
                    self.direction = RIGHT
                if event.key == pg.K_UP:
                    self.direction = UP
                if event.key == pg.K_DOWN:
                    self.direction = DOWN
                if event.key == pg.K_ESCAPE:
                    self.episode_over = True

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
                         ((self.nr_cells[0] // 2 - 3) * self.size_cells[0],
                          (self.nr_cells[1] // 2 - 1) * self.size_cells[1]))
        text_score = font.render(f"Score: {self.get_length_worn()}",
                                 True, (255, 255, 255))
        self.screen.blit(text_score,
                         ((self.nr_cells[0] // 2 - 3) * self.size_cells[0],
                          (self.nr_cells[1] // 2 + 1) * self.size_cells[1]))
        pg.display.flip()  # Update the full display
        self.clock.tick(0.5)
        self._print_grass()
        self.reset()
        self._render()

    def run(self):
        """Run the game in manual mode."""
        self.menu_active = False
        self.clock.tick(2)
        self.episode_over = False
        while not self.episode_over:
            self._check_event_()
            if not self.episode_over:
                _, self.episode_over = self._move()
            self._print_grass()
            self._render()
            pg.display.flip()
            self.clock.tick(2 + (self.get_length_worn() - 3) // 2)
        statistics = self.get_statistics()
        self._print_gameover()
        self.menu_active = True
        try:
            df = pd.read_csv(self.stats_manual)
            df = pd.concat([df, pd.DataFrame([statistics])], ignore_index=True)
            df.to_csv(self.stats_manual, index=False)
        except FileNotFoundError:
            print(f"Statistics file {self.stats_manual} not found.")

    def _auto_play(self):
        """Auto-play the game using the agent's policy."""
        self.menu_active = False
        self._autoplaying = True
        self.clock.tick(2)
        self.episode_over = False
        observation = self.get_observation()
        while not self.episode_over:
            self._check_event_()
            action, _ = self.agent.choose_action(observation)
            observation, _, terminated, truncated, _ = self.step(action)
            self._print_grass()
            self._render()
            pg.display.flip()
            self.clock.tick(10)
            if not self.episode_over:
                self.episode_over = terminated or truncated
        self._autoplaying = False
        self.menu_active = True
        stats = self.get_statistics()
        self._print_gameover()
        try:
            df = pd.read_csv(self.stats_auto)
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
            df.to_csv(self.stats_auto, index=False)
        except FileNotFoundError:
            print(f"Statistics file {self.stats_auto} not found.")

    def step(self, action):
        """Perform a step in the game with the given action.
        Args:
            action (int): The action to perform, represented as an integer.
        Returns:
            tuple: A tuple containing the observation, reward,
                    termination status, truncation status,
                    and additional info.
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
                self.terminated, self.truncated, {'moves': self.get_moves()})

    def _learn_game(self,
                    max_episodes=1000,
                    learn_type="Q_LEARNING",
                    FileNameModel="Model_Q.pt",
                    gpu_number=0):
        max_length = 3
        first_time = time.time()
        statistics = pd.DataFrame(columns=["red_apples",
                                           "green_apples",
                                           "score",
                                           "moves"])
        self.menu_active = False
        self.episode_over = False
        self._learning_game = True
        self.agent.set_epsilon(0.9)
        for i in range(max_episodes):
            observation, info = self.reset()
            game_over = False
            while not game_over:
                # agent policy that uses the observation and info
                action, _ = self.agent.choose_action(observation)
                (next_observation,
                 reward,
                 terminated,
                 truncated,
                 info) = self.step(action)
                self.agent.store_experience(observation,
                                            action,
                                            reward.value,
                                            next_observation,
                                            terminated or truncated)
                self.agent.train_single_step(observation,
                                             action,
                                             reward.value,
                                             next_observation,
                                             terminated or truncated)
                observation = next_observation
                if platform.system() == "Windows":
                    os.system("cls")
                else:
                    os.system('clear')
                max_length = max(self.get_length_worn(), max_length)
                print(f"Episode {i + 1}/{max_episodes}", end="\t")
                print(f"-Length {self.get_length_worn()}", end="\t")
                print(f"-Max_length {max_length}", end="\t")
                print(f"-Action: {Action(action).get_action_name()}",
                      end="\t")
                print(f"Reward: {reward}")
                print(f"-Epsilon: {self.agent.epsilon:.4f}", end="\t")
                print(f"-Terminated: {terminated}", end="\t")
                print(f"-Truncated: {truncated}", end="\t")
                print(f"- Moves: {info['moves']}")
                num_Chars = 50
                percent = 100 * (i + 1) / max_episodes
                filled = int(num_Chars * percent // 100)
                bar = '█' * filled + '-' * (num_Chars - filled)
                print('', end='\r')
                time_elapsed = (time.time() - first_time) / 3600
                print(f"Time elapsed: {time_elapsed:.2f} hours,",
                      end="\t")
                print(f"|{bar}| {percent:.2f}%", end="\t")
                time_left = ((max_episodes - i - 1) *
                             (time.time() - first_time) /
                             (i + 1) / 3600)
                print(f"- time left: {time_left:.2f} hours")
                self._print_grass()
                self._render()
                pg.display.flip()
                # self.print_map_in_shell()
                self.clock.tick(10)
                self._check_event_()
                if not game_over:
                    game_over = terminated or truncated
                elif self.episode_over:
                    game_over = True
            if self.episode_over:
                break
            self.agent.train_all()
            statistics = pd.concat([statistics, pd.DataFrame([self.get_statistics()])], ignore_index=True)
        self.agent.set_epsilon(0.01)
        self.episode_over = False
        self._learning_game = False
        self.menu_active = True
        try:
            df = pd.read_csv(self.stats_learn)
            df = pd.concat([df, statistics], ignore_index=True)
            df.to_csv(self.stats_learn, index=False)
        except FileNotFoundError:
            print(f"Statistics file {self.stats_learn} not found.")

    def _select_mode(self):
        """Display the mode selection menu."""
        self._print_grass()
        self._render()
        self.menu_font = pg.font.Font(None, 48)
        self.info_font = pg.font.Font(None, 28)

        # Define button rectangles
        self.manual_button = pg.Rect(self.screen.get_width() // 2 - 100,
                                     180,
                                     200,
                                     50)
        self.auto_button = pg.Rect(self.screen.get_width() // 2 - 100,
                                   250,
                                   200,
                                   50)
        self.learn_button = pg.Rect(self.screen.get_width() // 2 - 100,
                                    320,
                                    200,
                                    50)
        self.stats_button = pg.Rect(self.screen.get_width() // 2 - 100,
                                    390,
                                    200,
                                    50)

        while self.menu_active:
            # Title
            title = self.menu_font.render("Select Mode", True, (255, 255, 255))
            self.screen.blit(title,
                             ((self.screen.get_width() // 2 -
                               title.get_width() // 2),
                              100))

            # Draw buttons
            pg.draw.rect(self.screen, (70, 130, 180), self.manual_button)
            if self.show_autoplay:
                pg.draw.rect(self.screen, (34, 139, 34), self.auto_button)
            pg.draw.rect(self.screen, (255, 69, 0), self.learn_button)
            pg.draw.rect(self.screen, (84, 110, 122), self.stats_button)
            # Buttons text
            manual_text = self.info_font.render("Manual Mode",
                                                True,
                                                (255, 255, 255))
            self.screen.blit(manual_text,
                             (self.manual_button.x + 30,
                              self.manual_button.y + 10))
            if self.show_autoplay:
                auto_text = self.info_font.render("Auto Mode",
                                                  True,
                                                  (255, 255, 255))
                self.screen.blit(auto_text,
                                 (self.auto_button.x + 40,
                                  self.auto_button.y + 10))
            learn_text = self.info_font.render("Learn Mode",
                                               True,
                                               (255, 255, 255))
            self.screen.blit(learn_text,
                             (self.learn_button.x + 40,
                              self.learn_button.y + 10))
            learn_text = self.info_font.render("Statistics",
                                               True,
                                               (255, 255, 255))
            self.screen.blit(learn_text,
                             (self.stats_button.x + 40,
                              self.stats_button.y + 10))
            pg.display.flip()
            self._check_event_()
        pg.quit()

    def _check_event_statistics_(self):
        """Check events in the statistics menu."""
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                pg.quit()
                exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.menu_stats_active = False
                    return
            if event.type == pg.MOUSEBUTTONDOWN:
                if self.manual_stats_button.collidepoint(pg.mouse.get_pos()):
                    self._show_statistics(type="manual")
                    return
                elif self.auto_stats_button.collidepoint(pg.mouse.get_pos()):
                    self._show_statistics(type="auto")
                    return
                elif self.learn_stats_button.collidepoint(pg.mouse.get_pos()):
                    self._show_statistics(type="learn")
                    return
        return
    
    def _select_statistics(self):
        self._print_grass()
        self._render()
        self.menu_font = pg.font.Font(None, 48)
        self.info_font = pg.font.Font(None, 28)

        # Define button rectangles
        self.manual_stats_button = pg.Rect(self.screen.get_width() // 2 - 150,
                                     180,
                                     300,
                                     50)
        self.auto_stats_button = pg.Rect(self.screen.get_width() // 2 - 150,
                                   240,
                                   300,
                                   50)
        self.learn_stats_button = pg.Rect(self.screen.get_width() // 2 - 150,
                                    300,
                                    300,
                                    50)
        self.menu_stats_active = True
        while self.menu_stats_active:
            # Title
            title = self.menu_font.render("Select Statsistics to show", True, (255, 255, 255))
            self.screen.blit(title,
                             ((self.screen.get_width() // 2 -
                               title.get_width() // 2),
                              100))

            # Draw buttons
            pg.draw.rect(self.screen, (70, 130, 180), self.manual_stats_button)
            pg.draw.rect(self.screen, (34, 139, 34), self.auto_stats_button)
            pg.draw.rect(self.screen, (255, 69, 0), self.learn_stats_button)
            # Buttons text
            manual_text = self.info_font.render("Manual play Statistics",
                                                True,
                                                (255, 255, 255))
            self.screen.blit(manual_text,
                             (self.manual_stats_button.x + 30,
                              self.manual_stats_button.y + 10))
            auto_text = self.info_font.render("Auto play Statistics",
                                                True,
                                                (255, 255, 255))
            self.screen.blit(auto_text,
                                (self.auto_stats_button.x + 40,
                                self.auto_stats_button.y + 10))
            learn_text = self.info_font.render("Learn play Statistics",
                                               True,
                                               (255, 255, 255))
            self.screen.blit(learn_text,
                             (self.learn_stats_button.x + 40,
                              self.learn_stats_button.y + 10))
            pg.display.flip()
            self._check_event_statistics_()
        self._print_grass()
        self._render()

    def _show_statistics(self,type="manual"):
        """Display the statistics of the game."""

        if type == "manual" and self.stats_manual is not None:
            try:
                df = pd.read_csv(self.stats_manual)
                self.show_graphics(df)
            except FileNotFoundError:
                print(f"Statistics file {self.stats_manual} not found.")
        if type == "auto" and self.stats_auto is not None:
            try:
                df = pd.read_csv(self.stats_auto)
                self.show_graphics(df)
            except FileNotFoundError:
                print(f"Statistics file {self.stats_auto} not found.")
        if type == "learn" and self.stats_learn is not None:
            try:
                df = pd.read_csv(self.stats_learn)
                self.show_graphics(df)
            except FileNotFoundError:
                print(f"Statistics file {self.stats_learn} not found.")
        return

    def show_graphics(self, df):
        
        # 1. Create the matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        episodes = range(len(df))  # X-axis: episode indices

        ax.plot(episodes, df['score'], label='Score')
        ax.plot(episodes, df['green_apples'], label='Green Apples')
        ax.plot(episodes, df['red_apples'], label='Red Apples')
        ax.plot(episodes, df['moves'], label='Moves')
        ax.set_title('Length of Snake Over Episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        #ax.show()
        # 2. Render the plot to a canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        # 3. Convert to a numpy RGB array
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)

        raw_data = canvas.buffer_rgba()  # not rgba
        image = np.frombuffer(raw_data, dtype=np.uint8).reshape(width,height,4)

        # 4. Convert to Pygame surfaceyter
        surface = pg.image.frombuffer(image.tobytes(), (width, height), "RGBA")
        self.screen.blit(surface, (0, 0))

        plt.close(fig)
        pg.display.flip()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self._print_grass()
                        self._render()
                        pg.display.flip()
                        return
            self.clock.tick(60)
