import numpy as np  
from activation import Activation

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.input = None
        self.output = None
        self.delta = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input):
        self.input = input
        self.output = self.activation.forward(np.dot(input, self.weights) + self.bias)
        return self.output

    def backward(self, delta):
        self.delta = delta * self.activation.backward(self.output)
        self.delta_weights = np.dot(self.input.T, self.delta)
        self.delta_bias = np.sum(self.delta, axis=0)
        return np.dot(self.delta, self.weights.T)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias

    def __str__(self):
        return f"Layer: {self.input_dim} -> {self.output_dim} Activation: {self.activation}"