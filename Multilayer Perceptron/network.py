import numpy as np
from activation import Activation
from layer import Layer

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, epochs, learning_rate):
        for _ in range(epochs):
            for i in range(len(x)):
                output = self.forward(x[i])
                delta = output - y[i]
                self.backward(delta)
                self.update(learning_rate)

    def predict(self, x):
        return np.array([self.forward(x_) for x_ in x])

    def __str__(self):
        return "\n".join([str(layer) for layer in self.layers])