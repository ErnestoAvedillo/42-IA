from env_snake import EnvSnake
from dqn_agent import DQNAgent
import time

env = EnvSnake(Nr_cells=[10, 10])
agent = DQNAgent(state_shape=env.observation_space,
                 num_actions=env.action_space)
episode_over = False
print("Training completed.")
agent.load_model("dqn_snake_model.json")

observation, info = env.reset()
while not episode_over:
    action = agent.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.print_map_in_shell()
    episode_over = terminated or truncated
    time.sleep(1)