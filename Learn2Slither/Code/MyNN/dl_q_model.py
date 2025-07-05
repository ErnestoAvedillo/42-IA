from neural_network_class.network import Network
import numpy as np
# import tensorflow as tf

# def CreateDLQModel(state_shape, nr_actions):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(shape=state_shape))
#     model.add(tf.keras.layers.Dense(64, activation='relu'))
#     model.add(tf.keras.layers.Dense(32, activation='relu'))
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(nr_actions, activation='linear'))  # Linear activation for Q-values
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     return model
# 
# def ModelFit(model: tf.keras.Model, X: np.array, Y: np.array):
#     model.fit(X, Y, epochs=1, verbose=0)  # Train for one epoch without verbose output
    # Optionally, you can return the history if needed
    # return model.history.history


def CreateDLQModel(state_shape, nr_actions):
    model = Network(normalize=True)
    model.add_layer(layer_type='input', data_shape=state_shape)
    model.add_layer(layer_type='dense', input_shape=64, activation='relu')
    model.add_layer(layer_type='dense', input_shape=32, activation='relu')
    model.add_layer(layer_type='dense', input_shape=16, activation='relu')
    model.add_layer(layer_type='dense', input_shape=4, activation='linear')
    return model

def ModelFit(model: Network, X: np.array, Y: np.array):
    model.fit(X, Y, optimizer='adam')
