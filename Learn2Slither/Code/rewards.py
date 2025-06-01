from enum import Enum
from collisions import Collision


class Reward(Enum):
    # No reward
    NONE = 1
    # Reward for eating a red apple
    RED_APPLE = -2
    # Reward for eating a green apple
    GREEN_APPLE = 10
    # Penalty for hitting a wall or body
    WALL_PENALTY = -7
    # Penalty for hitting the snake's own body
    BODY_PENALTY = -7

    def get_reward(self, collision: Collision):
        match collision:
            case Collision.NONE:
                return Reward.NONE.value
            case Collision.RED_APPLE:
                return Reward.RED_APPLE.value
            case Collision.GREEN_APPLE:
                return Reward.GREEN_APPLE.value
            case Collision.WALL | Collision.BODY:
                return Reward.WALL_PENALTY.value
            case Collision.BODY:
                return Reward.BODY_PENALTY.value
            case _:
                return 0

    def __str__(self):
        return self.name.replace("_", " ").title()

    def __repr__(self):
        return f"Reward.{self.name}"
