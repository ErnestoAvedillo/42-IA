from env_snake import EnvSnake
from dqn_agent_keras import DQNAgent
from actions import Action
import time
import sys
from count_rewards import CountRewards

if len(sys.argv) > 1:
    try:
        File_Name = sys.argv[1]
    except ValueError as e:
        print(f"Error: {e}. Defaulting to 'dqn_snake_model.json'.")
        File_Name = "dqn_snake_model.json"

env = EnvSnake(Nr_cells=[10, 10])
agent = DQNAgent(state_shape=len(env.observation_space),
                 num_actions=env.action_space)
episode_over = False
print("Training completed.")
agent.load_model(File_Name)
print(f"Model loaded from {File_Name}")
# initialize counter of rewards for each action
rewards = [0, 0, 0, 0, 0, 0]

observation, info = env.reset()
total_rewards = CountRewards()
while not episode_over:
    action, is_aleatory = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.print_map_in_shell()
    episode_over = terminated or truncated
    total_rewards.add_reward(reward)
    print(f"Action: {Action(action).get_action_name()}", end="\t")
    print(f"Aleatory {is_aleatory}", end="\t")
    print(f"Reward: {reward.value}", end="\t")
    print(f"Episode Over: {episode_over}")
    for my_reward, name, value in total_rewards.total_rewards():
        print(f"{name}, {value} --", end='\t')
    print("")
    time.sleep(1)
