from actions import Action
from enum import Enum
from random import choice


class Directions(Enum):
    # Directions for the snake
    UP = [0, -1]
    DOWN = [0, 1]
    LEFT = [-1, 0]
    RIGHT = [1, 0]

    def get_direction(action):
        """Get the direction corresponding to the action where
        each action corresponds to this mapping in case is an integer:
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT
        """
        if isinstance(action, int):
            directions = list(Directions)
            if 0 <= action < len(directions):
                return directions[action].value
        elif isinstance(action, Action):
            action = Action.get_action_index(action)
            return Directions.get_direction(action)
        elif isinstance(action, Directions):
            # If action is already a Directions enum, return its value
            return action.value
        else:
            raise ValueError(f"Invalid action index: {action}")

    def get_random_direction():
        """Return a random direction from the Directions enum."""
        return Directions(choice(list(Directions))).value
