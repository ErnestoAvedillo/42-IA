import numpy as np
import torch
import random
from dl_q_model_keras import DLQModel
import joblib
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
MAX_MOVES = 1000                # Max number of moves per episode


class DQNAgent():
    def __init__(self, state_shape, num_actions, filename=None,
                 learning_type="Q_LEARNING", gpu_number=0):
        """ Initialize the DQN agent with the given parameters.
        Args:
            state_shape (tuple):
                Shape of the state space.
            num_actions (int):
                Number of possible actions.
            filename (str, optional):
                Filename to save/load the model. Defaults to None.
            learning_type (str, optional):
                Type of learning algorithm.
                Defaults to "Q_LEARNING", alterantive "SARSA".
            gpu_number (int, optional):
                GPU number to use for the neural network.
                Defaults to 0.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.policy_model = DLQModel((state_shape), num_actions, gpu_number)
        self.target_model = DLQModel((state_shape), num_actions, gpu_number)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.load_type = learning_type
        if self.load_type not in ["Q_LEARNING", "SARSA"]:
            raise ValueError("Invalid learning type. Valid "
                             "'Q_LEARNING' or 'SARSA'.")
        self.epsilon = EPSILON_START
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        if filename is not None:
            self.filename = filename
            if os.path.isfile(filename):
                self.load_model(filename)
            else:
                print(f"File {filename} does not exist."
                      "Starting with a new model.")
                self.filename = None
        self.training_steps = TARGET_UPDATE_FREQ
        self.save_srteps = TARGET_SAVE_FREQ
        self.moves = 0
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
        self.moves += 1
        if self.moves >= MAX_MOVES:
            self.truncated = True
        return output, is_aleatory

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Update epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        # Predict Q-values for current and next states
        with torch.no_grad():
            target_q_values = self.policy_model.forward(states).cpu().numpy()
            next_q_values = self.target_model.forward(next_states).cpu().numpy()

            for i in range(BATCH_SIZE):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    if self.load_type == "Q_LEARNING":
                        target_q_values[i][actions[i]] = (
                            (1 - AGENT_LEARNING_RATE) * (
                                target_q_values[i][actions[i]]
                            ) +
                            AGENT_LEARNING_RATE * (
                                rewards[i] + GAMMA *
                                np.max(next_q_values[i])
                            )
                        )
                    elif self.load_type == "SARSA":
                        target_q_values[i][actions[i]] = (
                            (1 - AGENT_LEARNING_RATE) * (
                                target_q_values[i][actions[i]]
                            ) + AGENT_LEARNING_RATE * (
                                rewards[i] + GAMMA *
                                next_q_values[i][actions[i]]
                                )
                            )
        if self.training_steps == 0:
            # Train the policy model
            self.policy_model.fit(states, target_q_values, epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  learning_rate=LEARNING_RATE)
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.training_steps = TARGET_UPDATE_FREQ
        else:
            self.training_steps -= 1
        del target_q_values
        if self.filename is not None and self.save_srteps == 0:
            self.save_model(self.filename)
            self.save_srteps = TARGET_SAVE_FREQ
        else:
            self.save_srteps -= 1

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
        model = self.get_model()
        joblib.dump(model, filename)

    def load_model(self, filename):
        model = joblib.load(filename)
        self.set_model(model)
