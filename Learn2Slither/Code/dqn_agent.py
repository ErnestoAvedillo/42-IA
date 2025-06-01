import numpy as np
import random
from dl_q_model import CreateDLQModel
import json

# --- Hyperparameters ---
GAMMA = 0.50                    # Discount factor
LEARNING_RATE = 0.001           # Learning rate for the neural network
REPLAY_BUFFER_SIZE = 100_000    # Max experiences in replay buffer
BATCH_SIZE = 32                # Number of experiences to sample for training
EPSILON_START = 1.0             # Initial exploration rate
EPSILON_END = 0.01              # Minimum exploration rate
EPSILON_DECAY = 0.9995          # Rate at which epsilon decays per episode
TARGET_UPDATE_FREQ = 100        # How often to update the target network
                                # (in training steps)
TARGET_SAVE_FREQ = 1000         # How often to save the target model

class DQNAgent():
    def __init__(self, state_shape, num_actions,filename=None):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.policy_model = CreateDLQModel((BATCH_SIZE, state_shape), num_actions)
        self.target_model = CreateDLQModel((BATCH_SIZE, state_shape), num_actions)
        # model = self.policy_model.get_model()
        # self.target_model.set_model(model)
        self.target_model.set_weights(self.policy_model.get_weights())
        self.epsilon = EPSILON_START
        self.replay_buffer = []
        self.filename = filename
        self.training_steps = TARGET_UPDATE_FREQ
        self.save_srteps = TARGET_SAVE_FREQ

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            # for linear activation, the output is a vector of Q-values
            # q_values = self.policy_model.forward(np.array(state))
            current_state = np.array(state)[np.newaxis, np.newaxis, ...]
            q_values = self.policy_model.predict(current_state, verbose=0)
            # in case just take directly the max Q-value
            return np.argmax(q_values)
            # in case we want to sample aleatory an action based on the Q-values weights
            # for softmax activation, the output is a probability distribution
            # q_values = self.policy_model.forward(np.array(state).reshape(1, -1))
            # return np.random.choice(q_values.shape[1], p=q_values[0] / np.sum(q_values[0]))

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

        # Predict Q-values for current and next states
        # current_q_values = self.policy_model.forward(states)
        # current_q_values = self.policy_model.forward(states)
        current_q_values = self.policy_model.predict(states[np.newaxis, ...])
        next_q_values = self.target_model.predict(next_states[np.newaxis, ...])
        current_q_values = current_q_values.reshape(BATCH_SIZE, self.num_actions)
        next_q_values = next_q_values.reshape(BATCH_SIZE, self.num_actions)
        # Update Q-values using Bellman equation
        target_q_values = np.copy(current_q_values)
        for i in range(BATCH_SIZE):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = (rewards[i] + GAMMA *
                                           np.max(next_q_values[i]))

        # Train the policy model
        # for softmax activation, we need to convert Q-values to probabilities
        # q_values_copy = np.zeros_like(q_values)
        # max_indices = q_values.argmax(axis=1)
        # q_values_copy[np.arange(BATCH_SIZE), max_indices] = 1
        # self.policy_model.fit(states, q_values_copy, epochs=5, verbose=0)
        # for linear activation, we can use the Q-values directly
#        print("Training with states shape:", states, "and q_values shape:", target_q_values)
        self.policy_model.fit(states[np.newaxis, ...], target_q_values[np.newaxis, ...], epochs=5, verbose=0)

        # Update epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        
        if self.training_steps == 0:
            # model = self.policy_model.get_model()
            # self.target_model.set_model(model)
            self.target_model.set_weights(self.policy_model.get_weights())
            self.training_steps = TARGET_UPDATE_FREQ
        else:
            self.training_steps -= 1
        
        if self.filename is not None and self.save_srteps == 0:
            self.save_model(self.filename)
            self.save_srteps = TARGET_SAVE_FREQ
        else:
            self.save_srteps -= 1

    def get_model(self):
        model = {}
        model["policy_model"] = self.policy_model.get_model()
        model["target_model"] = self.target_model.get_model()
        model["epsilon"] = self.epsilon
        model["state_shape"] = self.state_shape
        model["num_actions"] = self.num_actions
        return model

    def set_model(self, model):
        self.policy_model.set_model(model["policy_model"])
        self.target_model.set_model(model["target_model"])
        self.epsilon = model["epsilon"]
        self.state_shape = model["state_shape"]
        self.num_actions = model["num_actions"]

    def save_model(self, filename):
        model = self.get_model()
        with open(filename, "w") as archivo:
            json.dump(model, archivo)

    def load_model(self, filename):
        with open(filename, "r") as archivo:
            model = json.load(archivo)
        self.set_model(model)