from enum import Enum


class Collision(Enum):
    # No collision
    NONE = 0
    # Collision with red apple
    RED_APPLE = 1
    # Collision with green apple
    GREEN_APPLE = 2
    # Collision with wall or body
    WALL = 3
    # Collision with body
    BODY = 4
    # No collision, Green appla in Row or col
    IS_THE_WAY = 5
    # No collision, repeated position
    REPEATED_POSITION = 6
    # No collision, alligned with green apple
    IS_ALLIGNED_WITH_GREEN_APPLE = 7

    def __str__(self):
        return self.name.replace("_", " ").title()

    def __repr__(self):
        return f"Collision.{self.name}"
