from enum import Enum
from collisions import Collision

DICCTIONARY = {
    Collision.NONE: "No collision",
    Collision.IS_THE_WAY: "Is the way",
    Collision.RED_APPLE: "Red apple",
    Collision.GREEN_APPLE: "Green apple",
    Collision.WALL: "Wall",
    Collision.BODY: "Body",
    Collision.REPEATED_POSITION: "Repeated position"
}


class Reward(Enum):
    # No reward
    NONE = -0.1
    # REWARD ON THE WAY
    IS_THE_WAY = 100
    # Reward for being on the way to a green apple
    IS_ALLIGNED_WITH_GREEN_APPLE = 50
    # Penalty for repeated position
    IS_REPEATED_POSITION = -200
    # Green apple reward on th way
    RED_APPLE = -50
    # Reward for eating a green apple
    GREEN_APPLE = 500
    # Penalty for hitting a wall or body
    WALL_PENALTY = -400
    # Penalty for hitting the snake's own body
    BODY_PENALTY = -450

    def get_reward(self, collision: Collision):
        match collision:
            case Collision.NONE:
                return Reward.NONE.value
            case Collision.IS_THE_WAY:
                return Reward.IS_THE_WAY.value  # No reward for being on the way
            case Collision.IS_ALLIGNED_WITH_GREEN_APPLE:
                return Reward.IS_ALLIGNED_WITH_GREEN_APPLE.value
            case Collision.RED_APPLE:
                return Reward.RED_APPLE.value
            case Collision.GREEN_APPLE:
                return Reward.GREEN_APPLE.value
            case Collision.WALL:
                return Reward.WALL_PENALTY.value
            case Collision.BODY:
                return Reward.BODY_PENALTY.value
            case Collision.REPEATED_POSITION:
                return Reward.IS_REPEATED_POSITION.value
            case _:
                return 0

    @classmethod
    def get_len(cls):
        return len(cls)
    
    def __str__(self):
        return self.name.replace("_", " ").title()

    def __repr__(self):
        return f"Reward.{self.name}"

    def get_reward_name(self, collision: Collision):
        """Get the name of the reward based on the collision."""
        if collision not in DICCTIONARY.keys():
            raise ValueError(f"Invalid collision value: {collision}")
        return (DICCTIONARY[collision])