from enum import Enum
from random import choice

NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class Action(Enum):
    # Actions for the snake
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def return_random_action():
        """Return a random action from the Action enum."""
        return Action(choice(list(Action)))

    def get_action(action: int):
        """Get the action corresponding to the action int."""
        return Action(action) if action in range(len(Action)) else None

    def get_len_actions():
        return len(Action)

    def get_action_name(self, action: int=None):
        """Print the name of the action."""
        return (NAMES[self.value])
