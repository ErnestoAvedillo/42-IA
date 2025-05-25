from actions import Action
from enum import Enum
from random import choice


class Directions(Enum):
    # Directions for the snake
    UP = [0, -1]
    DOWN = [0, 1]
    LEFT = [-1, 0]
    RIGHT = [1, 0]

    def get_direction(action: Action):
        """Get the direction corresponding to the action."""
        return Directions[action.value] if action in Action else None

    def get_random_direction():
        """Return a random direction from the Directions enum."""
        return Directions(choice(list(Directions))).value
