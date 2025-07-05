from env_snake import EnvSnake
from dqn_agent_keras import DQNAgent
from actions import Action
from rewards import Reward
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import platform
import os
import argparse
import time
from count_rewards import CountRewards

NUM_EPISODES = 500000       # Total episodes to train for


if __name__ == "__main__":
    parser_description = "Script to train a DQN agent to play the Snake game."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('-f',
                        '--file_model',
                        type=str,
                        nargs='?',
                        default='dqn_snake_model.joblib',
                        help='File name to save the model.')
    parser.add_argument('-g',
                        '--gpu_nr',
                        type=int,
                        nargs='?',
                        help='GPU number where to xecute the neural network.')
    parser.add_argument('--history_lengths',
                        type=str,
                        nargs='?',
                        default='lengths.csv',
                        help='File to save the hystoric lengths for games.')
    parser.add_argument('-e',
                        '--episodes',
                        type=int,
                        nargs='?',
                        default=1000,
                        help='Number of maximum episodes to repeat.')
    args = parser.parse_args()

    # Check if file name is provided,
    if not args.file_model:
        print("No file name provided. Defaulting to 'dqn_snake_model.joblib'.")
        File_Name = "dqn_snake_model.joblib"
    else:
        File_Name = args.file_model
    if not args.gpu_nr:
        print("No GPU number assigned. Default number taken 0")
        gpu_number = 0
    else:
        gpu_number = args.gpu_nr
    if not args.history_lengths:
        print("No history filename given. Default name taken lengths.csv")
        filename_lengths = "lengths.csv"
    else:
        filename_lengths = args.history_lengths
    if not args.episodes:
        print("")
        max_episodes = NUM_EPISODES
    else:
        max_episodes = args.episodes

env = EnvSnake(Nr_cells=[10, 10])
agent = DQNAgent(state_shape=len(env.observation_space),
                 num_actions=env.action_space,
                 filename=File_Name,
                 gpu_number=gpu_number)
# initialize counter of rewards for each action
rewards = [0 for _ in range(Reward.get_len() + 1)]
max_length = 3
first_time = time.time()
lengths = []
total_rewards = CountRewards()
for i in range(max_episodes):
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        # agent policy that uses the observation and info
        action, _ = agent.choose_action(observation)
        (next_observation,
         reward,
         terminated,
         truncated,
         info) = env.step(action)
        agent.store_experience(observation,
                               action,
                               reward.value,
                               next_observation,
                               terminated or truncated)
        agent.train_single_step(observation,
                                action,
                                reward.value,
                                next_observation,
                                terminated or truncated)
        observation = next_observation
        total_rewards.add_reward(reward)
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system('clear')
        env.print_map_in_shell()
        episode_over = terminated or truncated
        max_length = max(env.get_length_worn(), max_length)
        print(f"Episode {i + 1}/{max_episodes}", end="\t")
        print(f"- Length {env.get_length_worn()}", end="\t")
        print(f"- Max_length {max_length}", end="\t")
        print(f"- Action: {Action(action).get_action_name()}")
        print(f"Epsilon: {agent.epsilon:.4f}", end="\t")
        print(f"- Terminated: {terminated}", end="\t")
        print(f"- Truncated: {truncated}", end="\t")
        print(f"- Moves: {info['moves']}", end="\t")
        print(f"- Reward: {reward:.20}")
        print("Rewards historic:", end="\t")
        for my_reward, name, value in total_rewards.total_rewards():
            print(f"{name}: {value} --", end="\t")
        num_Chars = 50
        percent = 100 * (i + 1) / max_episodes
        filled = int(num_Chars * percent // 100)
        bar = '█' * filled + '-' * (num_Chars - filled)
        print('', end='\r')
        print(f"Time elapsed: {(time.time() - first_time) / 3600:.2f} hours,",
              end="\t")
        print(f"|{bar}| {percent:.2f}%", end="\t")
        time_left = ((max_episodes - i - 1) *
                     (time.time() - first_time) /
                     (i + 1) / 3600)
        print(f"- time left: {time_left:.2f} hours")
    agent.train_all()
    lengths.append(env.get_length_worn())
    pd.DataFrame(lengths, columns=["length"]).to_csv(filename_lengths,
                                                     index=False)
episode_over = False
lengths = np.array(lengths)
plt.plot(lengths, label='Length of Snake')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Length of Snake Over Episodes')
plt.legend()
plt.show()

print("Training completed.")

agent.save_model(File_Name)

observation, info = env.reset()
while not episode_over:
    action, is_aleatory = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.print_map_in_shell()
    episode_over = terminated or truncated
    time.sleep(0.25)
    print(f"Action: {Action(action).get_action_name()}", end="\t")
    print(f"- Length {env.get_length_worn()}", end="\t")
    print(f"- Aleatory {is_aleatory}", end="\t")
    print(f"- Reward: {reward.value}", end="\t")
    print(f"- Episode Over: {episode_over}")
