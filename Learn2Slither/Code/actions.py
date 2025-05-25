from enum import Enum
from random import choice


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
