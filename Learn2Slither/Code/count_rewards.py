from rewards import Reward

class CountRewards:
    def __init__(self):
        self.rewards = Reward.get_list_rewards()
        self.reward_counter = {reward: 0 for reward in self.rewards}

    def add_reward(self, reward):
        if reward not in self.rewards:
            raise ValueError(f"Invalid reward type: {reward}")
        self.reward_counter[reward] += 1

    def get_count(self, reward):
        if reward not in self.rewards:
            raise ValueError(f"Invalid reward type: {reward}")
        return self.reward_counter[reward]

    def total_rewards(self):
        return zip(Reward.get_list_rewards(), Reward.get_reward_names(), self.reward_counter.values())
