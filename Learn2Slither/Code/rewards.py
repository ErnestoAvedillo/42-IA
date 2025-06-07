from enum import Enum
from collisions import Collision


class Reward(Enum):
    # No reward
    NONE = -0.1
    # Green apple reward on th way
    RED_APPLE = -5
    # Reward for eating a green apple
    GREEN_APPLE = 400
    # Penalty for hitting a wall or body
    WALL_PENALTY = -100
    # Penalty for hitting the snake's own body
    BODY_PENALTY = -200

    def get_reward(self, collision: Collision):
        match collision:
            case Collision.NONE:
                return Reward.NONE.value
            case Collision.RED_APPLE:
                return Reward.RED_APPLE.value
            case Collision.GREEN_APPLE:
                return Reward.GREEN_APPLE.value
            case Collision.WALL:
                return Reward.WALL_PENALTY.value
            case Collision.BODY:
                return Reward.BODY_PENALTY.value
            case _:
                return 0

    def __str__(self):
        return self.name.replace("_", " ").title()

    def __repr__(self):
        return f"Reward.{self.name}"
