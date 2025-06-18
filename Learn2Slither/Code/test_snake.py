from env_snake import EnvSnake
from dqn_agent_keras import DQNAgent
from actions import Action
import time
import sys


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
while not episode_over:
    action, is_aleatory = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.print_map_in_shell()
    episode_over = terminated or truncated
    match reward:
        case 0:  # Reward.NONE.value
            rewards[0] += 1
        case 1:  # Reward.RED_APPLE.value
            rewards[1] += 1
        case 2:  # Reward.GREEN_APPLE.value
            rewards[2] += 1
        case 3:  # Reward.WALL_PENALTY.value
            rewards[3] += 1
        case 4:  # Reward.BODY_PENALTY.value
            rewards[4] += 1
        case _:
            rewards[5] += 1  # Unhandled reward case
    print(f"Action: {Action(action).get_action_name()}, Aleatory {is_aleatory}, Reward: {reward}, Episode Over: {episode_over}")
    print(f"Rewards historic: NONE {rewards[0]}, RED {rewards[1]}, GREEN {rewards[2]}, WALL {rewards[3]}, BODY {rewards[4]}, UNHANDLED {rewards[5]}")
    time.sleep(1)