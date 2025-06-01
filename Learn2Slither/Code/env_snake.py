from motor_snake import MotorSnake
from directions import Directions
from actions import Action
from rewards import Reward
import numpy as np

DICTIONARY_OBSERVATION = {
    "O": 0,
    "W": 1,
    "H": 2,
    "S": 3,
    "G": 4,
    "R": 5
}


class EnvSnake(MotorSnake):
    """
    Environment for the Snake game, inheriting from MotorSnake and gym.Env.
    This class initializes the environment, handles actions, and provides
    observations.
    """
    def __init__(self, Nr_cells=[10, 10]):
        MotorSnake.__init__(self, Nr_cells)
        # gym.Env.__init__(self)
        self.action_space = Action.get_len_actions()
        # observation, _ =MotorSnake.reset()
        self.action = Action.return_random_action()
        self.truncated = False
        self.terminated = False
        self.observation_space =  len(self.get_observation())
        # self.reset()

    def reset(self, seed=None, options=None):
        MotorSnake.reset(self)
        self.observation_space = self.get_observation()
        self.terminated = False
        return self.observation_space, {}

    def step(self, action):
        # convert from np.int64 to int
        action = int(action)
        self.direction = Directions.get_direction(action)
        collision, self.terminated = self._move()
        self._create_map()
        self.reward = Reward.get_reward(self, collision)
        return (self.get_observation(), self.reward,
                self.terminated, self.truncated, {})

    def get_observation(self):
        if len(self.worn) == 0:
            head_col = 1
            head_raw = 1
        else:
            head_raw = self.worn[0][0]
            head_col = self.worn[0][1]
        
        raw = self.map[head_raw]
        col = [row[head_col] for row in self.map]
        raw = [DICTIONARY_OBSERVATION[cell] for cell in raw]
        col = [DICTIONARY_OBSERVATION[cell] for cell in col]
        observation = raw + col
        return observation
