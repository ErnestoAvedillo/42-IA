from env_snake import EnvSnake
from dqn_agent import DQNAgent
from actions import Action
import time

NUM_EPISODES = 1000    # Total episodes to train for


env = EnvSnake(Nr_cells=[10, 10])
agent = DQNAgent(state_shape=env.observation_space,
                 num_actions=env.action_space)
# agent = DQNAgent(state_shape=env.observation_space,
#                  num_actions=env.action_space, filename="dqn_snake_model.json")

for i in range(NUM_EPISODES):
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        # agent policy that uses the observation and info
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.store_experience(observation, action, reward, observation,
                               terminated or truncated)
        agent.train()
        env.print_map_in_shell()
        episode_over = terminated or truncated
        print(f"Episode {i + 1}/{NUM_EPISODES} - Action: {Action(action).get_action_name()}, Reward: {reward}, Epsilon: {agent.epsilon:.4f}, Episode Over: {episode_over}")
        # time.sleep(1)
episode_over = False
print("Training completed.")
agent.save_model("dqn_snake_model.json")

observation, info = env.reset()
while not episode_over:
    action = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.print_map_in_shell()
    episode_over = terminated or truncated
    time.sleep(1)