import numpy as np
import torch
import random
from dl_q_model_keras import DLQModel
from collections import deque
import os

# --- Hyperparameters ---
GAMMA = 0.95                    # Discount factor
LEARNING_RATE = 0.01            # Learning rate for the neural network
AGENT_LEARNING_RATE = 0.1       # Learning rate for the agent
REPLAY_BUFFER_SIZE = 1000       # Max experiences in replay buffer
BATCH_SIZE = 1000               # Number of experiences to sample for training
EPSILON_START = 1.0             # Initial exploration rate
EPSILON_END = 0.01              # Minimum exploration rate
EPSILON_DECAY = 0.999           # Rate at which epsilon decays per episode
TARGET_UPDATE_FREQ = 100        # How often to update the target network
TARGET_SAVE_FREQ = 100          # How often to save the target model
EPOCHS = 1                      # Number of epochs to train the model per batch


class DQNAgent():
    def __init__(self, state_shape, num_actions, filename=None,
                 gpu_number=0,
                 epsilon=EPSILON_START):
        """ Initialize the DQN agent with the given parameters.
        Args:
            state_shape (tuple):
                Shape of the state space.
            num_actions (int):
                Number of possible actions.
            filename (str, optional):
                Filename to save/load the model. Defaults to None.
            gpu_number (int, optional):
                GPU number to use for the neural network.
                Defaults to 0.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.policy_model = DLQModel((state_shape), num_actions, gpu_number)
        self.target_model = DLQModel((state_shape), num_actions, gpu_number)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.epsilon = epsilon
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        if filename is not None:
            self.filename = filename
            if os.path.isfile(filename):
                self.load_model(filename)
        else:
            self.filename = None
        self.save_steps = TARGET_SAVE_FREQ
        self.truncated = False

    def choose_action(self, state):
        is_aleatory = False
        if random.random() < self.epsilon:
            is_aleatory = True
            output = random.randrange(self.num_actions)
        else:
            current_state = np.array(state)
            q_values = self.policy_model.forward(current_state)
            output = torch.argmax(q_values).item()
        return output, is_aleatory

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_all(self):
        """ Train the agent using experiences stored in the replay buffer.
        This method samples a batch of experiences from the replay buffer,
        updates the Q-values, and trains the policy model.
        """
        if len(self.replay_buffer) > BATCH_SIZE:
            batch = random.sample(self.replay_buffer, BATCH_SIZE)
        else:
            batch = self.replay_buffer
        states, actions, rewards, next_states, dones = zip(*batch)
        self.train(states, actions, rewards, next_states, dones)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        if self.filename is not None and self.save_steps == 0:
            self.save_model(self.filename)
            self.save_steps = TARGET_SAVE_FREQ
        else:
            self.save_steps -= 1

    def train_single_step(self, state, action, reward, next_state, done):
        """ Train the agent for a single step using the given experience.
        Args:
            state (np.array): Current state of the environment.
            action (int): Action taken by the agent.
            reward (float): Reward received from the environment.
            next_state (np.array): Next state of the environment.
            done (bool): Whether the episode has ended.
        """
        # Update epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        self.train(state, action, reward, next_state, done)

    def train(self, states, actions, rewards, nxt_states, dones):
        """ Train the agent using the given experiences.
        Args:
            states (list): List of current states.
            actions (list): List of actions taken.
            rewards (list): List of rewards received.
            nxt_states (list): List of next states.
            dones (list): List of done flags indicating if the episode ended.
        """
        states = torch.tensor(states, dtype=torch.float32)
        nxt_states = torch.tensor(nxt_states, dtype=torch.float32)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            nxt_states = torch.unsqueeze(nxt_states, 0)
            rewards = torch.unsqueeze(rewards, 0)
            actions = torch.unsqueeze(actions, 0)
            dones = (dones,)
        # Predict Q-values for current and next states
        with torch.no_grad():
            target_q_values = self.policy_model.forward(states).cpu().numpy()
            next_q_values = self.target_model.forward(nxt_states).cpu().numpy()

            for i in range(states.shape[0]):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = (
                        (1 - AGENT_LEARNING_RATE) * (
                            target_q_values[i][actions[i]]
                        ) +
                        AGENT_LEARNING_RATE * (
                            rewards[i] + GAMMA *
                            np.max(next_q_values[i])
                        )
                    )

        self.policy_model.fit(states, target_q_values, epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              learning_rate=LEARNING_RATE)

    def get_model(self):
        model = {}
        model["policy_model"] = self.policy_model
        model["target_model"] = self.target_model
        model["epsilon"] = self.epsilon
        model["state_shape"] = self.state_shape
        model["num_actions"] = self.num_actions
        return model

    def set_model(self, model):
        self.policy_model = model["policy_model"]
        self.target_model = model["target_model"]
        self.epsilon = model["epsilon"]
        self.state_shape = model["state_shape"]
        self.num_actions = model["num_actions"]

    def save_model(self, filename):
        """ Save the model to a file.
        Args:
            filename (str): The name of the file to save the model.
        """
        self.policy_model.save(filename)

    def load_model(self, filename):
        """ Load the model from a file.
        Args:
            filename (str): The name of the file to load the model from.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        self.policy_model.load(filename)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        print(f"Model loaded from {filename}.")

    def set_model_name(self, filename):
        """ Set the filename for saving/loading the model.
        Args:
            filename (str): The name of the file to save/load the model.
        """
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        self.filename = filename
        print(f"Model name set to {self.filename}.")

    def set_epsilon(self, epsilon):
        """ Set the exploration rate (epsilon).
        Args:
            epsilon (float): The exploration rate.
        """
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1.")
        self.epsilon = epsilon
        print(f"Epsilon set to {self.epsilon}.")
