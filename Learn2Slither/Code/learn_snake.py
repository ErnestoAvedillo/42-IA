from env_snake import EnvSnake
from dqn_agent_keras import DQNAgent
#from dqn_agent import DQNAgent
from actions import Action
from rewards import Reward
import time
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import platform
import os
import argparse
import time


NUM_EPISODES = 500000       # Total episodes to train for


def Usage():
    print ("Usage:")
    print ("python learn_snake.py <type of learn> <model name to save")
    print ("Where:")
    print ("type to learn can be 'Q_LEARNING' or 'SARSA'")
    print ("Example:")
    print ("python learn_snake.py 'SARSA' 'model.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for the Snake game.')
    parser.add_argument('-l', '--learn_type', type=str, choices=['Q_LEARNING', 'SARSA'], help='Type of learning algorithm to use.')
    parser.add_argument('-f', '--file_name', type=str, nargs='?', default='dqn_snake_model.joblib', help='File name to save the model.')
    args = parser.parse_args()
    
    # Check if learning type is provided, otherwise default to 'SARSA
    if not args.learn_type:
        print("No learning type provided. Defaulting to 'Q_LEARNING'.")
        Learn_Type = "Q_LEARNING"
    elif args.learn_type not in ["Q_LEARNING", "SARSA"]:
        Usage()
        raise ValueError("Invalid learning type. Choose 'Q_LEARNING' or 'SARSA'.")
    else:
        print(f"Learning type set to: {args.learn_type}")
        Learn_Type = args.learn_type
    # Check if file name is provided, otherwise default to 'dqn_snake_model.joblib'
    if not args.file_name:
        print("No file name provided. Defaulting to 'dqn_snake_model.joblib'.")
        File_Name = "dqn_snake_model.joblib"
    else:
        File_Name = args.file_name

env = EnvSnake(Nr_cells=[10, 10])
agent = DQNAgent(state_shape=len(env.observation_space),
                 num_actions=env.action_space,learning_type=Learn_Type, filename=File_Name)
rewards = [0 for _ in range(Reward.get_len() + 1)] # initialize counter of rewards for each action
# agent = DQNAgent(state_shape=env.observation_space,
#                  num_actions=env.action_space, filename="dqn_snake_model.joblib")
max_length = 3 
first_time = time.time()
lengths = []
for i in range(NUM_EPISODES):
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        # agent policy that uses the observation and info
        action, _ = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.store_experience(observation, action, reward, observation,
                               terminated or truncated)
        match reward:
            case Reward.NONE.value:
                rewards[0] += 1
            case Reward.RED_APPLE.value:
                rewards[1] += 1
            case Reward.GREEN_APPLE.value:
                rewards[2] += 1
            case Reward.WALL_PENALTY.value:
                rewards[3] += 1
            case Reward.BODY_PENALTY.value:
                rewards[4] += 1
            case Reward.IS_THE_WAY.value:
                rewards[5] += 1  # Reward for being on the way
            case Reward.IS_ALLIGNED_WITH_GREEN_APPLE.value:
                rewards[6] += 1  # Reward for being aligned with green apple
            case Reward.IS_REPEATED_POSITION.value:
                rewards[7] += 1  # Penalty for repeated position
            case _:
                rewards[8] += 1  # Unhandled reward case
        agent.train()
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system('clear')
        #env.print_map_in_shell(clear=False)
        episode_over = terminated or truncated
        max_length = max (env.get_length_worn(), max_length)
        print(f"Episode {i + 1}/{NUM_EPISODES} \t - Length {env.get_length_worn()} \t - Max_length {max_length} \t - Action: {Action(action).get_action_name()}")
        print(f"Reward: {reward} \t- Epsilon: {agent.epsilon:.4f} \t - Terminated: {terminated} \t- Truncated: {truncated}\t - Moves: {info['moves']}")
        print(f"Rewards historic:\t- NONE {rewards[0]}\t- RED {rewards[1]}\t- GREEN {rewards[2]}\t - IS_THE_WAY {rewards[5]}\t - IS_ALLIGNED {rewards[6]}")
        print(f"\t\t\t- WALL {rewards[3]}\t- BODY {rewards[4]}\t- UNHANDLED {rewards[8]}\t - REPEATED_POSITION {rewards[7]}")
        num_Chars = 50
        percent = 100 * (i + 1) / NUM_EPISODES
        filled = int(num_Chars * percent // 100)
        bar = '█' * filled + '-' * (num_Chars - filled)
        print(f'', end='\r')
        print(f"Time elapsed: {(time.time() - first_time) / 3600 :.2f} hours,\t|{bar}| {percent:.2f}% \t - time left: {(NUM_EPISODES - i - 1) * (time.time() - first_time) / (i + 1) / 3600:.2f} hours")
        # time.sleep(1)
    lengths.append(env.get_length_worn())
    pd.DataFrame(lengths, columns=["length"]).to_csv("lengths.csv", index=False)
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
    time.sleep(1)
    print(f"Action: {Action(action).get_action_name()} \t - Length {env.get_length_worn()} - Aleatory {is_aleatory}, \t Reward: {reward}, \t Episode Over: {episode_over}")