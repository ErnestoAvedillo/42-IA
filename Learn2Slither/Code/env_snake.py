from motor_snake import MotorSnake
from directions import Directions
from actions import Action
from rewards import Reward

DICTIONARY_OBSERVATION = {
    "O": 1,
    "W": 2,
    "H": 3,
    "S": 4,
    "G": 5,
    "R": 6
}

MAX_MOVES = 1000  # Maximum number of moves before truncation


class EnvSnake(MotorSnake):
    """
    Environment for the Snake game, inheriting from MotorSnake and gym.Env.
    This class initializes the environment, handles actions, and provides
    observations.
    """
    def __init__(self, Nr_cells=[10, 10]):
        MotorSnake.__init__(self, Nr_cells)
        self.action_space = Action.get_len_actions()
        self.action = Action.return_random_action()
        self.reset()

    def reset(self, seed=None, options=None):
        MotorSnake.reset(self)
        self.observation_space = self.get_observation()
        self.truncated = False
        self.terminated = False
        return self.observation_space, {}

    def step(self, action):
        # convert from np.int64 to int
        action = int(action)
        self.direction = Directions.get_direction(action)
        collision, self.terminated = self._move()
        self._create_map()
        self.reward = Reward.get_reward(self, collision)
        if self.get_moves() >= MAX_MOVES:
            self.truncated = True
        return (self.get_observation(), self.reward,
                self.terminated, self.truncated, {"moves": self.get_moves()})

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
#        raw_col = numbered_raw + numbered_col
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
            if apple[0] < head_col - 1 and apple[1] == head_raw - 1:
                observation[4] = 1
            if apple[0] > head_col - 1 and apple[1] == head_raw - 1:
                observation[5] = 1
            if apple[0] == head_col - 1 and apple[1] > head_raw - 1:
                observation[6] = 1
            if apple[0] == head_col - 1 and apple[1] < head_raw - 1:
                observation[7] = 1
        # Check for coincidences in raw or col with red apples
        for apple in self.red_apples:
            if apple[0] < head_col - 1 and apple[1] == head_raw - 1:
                observation[8] = 1
            if apple[0] > head_col - 1 and apple[1] == head_raw - 1:
                observation[9] = 1
            if apple[0] == head_col - 1 and apple[1] > head_raw - 1:
                observation[10] = 1
            if apple[0] == head_col - 1 and apple[1] < head_raw - 1:
                observation[11] = 1
        return observation

    def get_length_worn(self):
        return len(self.worn)
