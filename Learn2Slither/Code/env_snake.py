from motor_snake import MotorSnake
from directions import Directions
from actions import Action
from rewards import Reward


class EnvSnake(MotorSnake):
    def __init__(self, x, y, Nr_cells=[10, 10]):
        super().__init__(x, y, Nr_cells)
        self.action = Action.return_random_action()

    def reset(self):
        super().reset()
        self.terminated = False
        return self.get_observation(), {}

    def step(self, action):
        self.direction = Directions.get_direction(action)
        self._move()
        self._create_map()
        self.reward = Reward.get_reward(self)
        return self.get_observation(), self.reward, self.terminated, {}

    def get_observation(self):
        raw = self.map[self.worn[0]]
        col = [row[self.worn[1]] for row in self.map]
        return raw + col
