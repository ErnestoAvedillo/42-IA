import numpy as np
from activation import Activation
from layer import Layer

class Network:
    def __init__(self,layers = [], x = None, y = None, learning_rate = 0.001):
        self.layers = layers
        self.x = None
        self.y_pred = None
        self.y = None
        self.learning_rate = None

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_layer(self, nodes, input_dim=None, activation="sigmoid"):
        if input_dim is None and len(self.layers) == 0:
            raise ValueError("First layer must have input_dim")
        else:
            input = input_dim if input_dim is not None else self.layers[-1].output_dim
        self.layers.append(Layer(input, nodes, activation))

    def forward(self):
        for layer in self.layers:
            self.y_pred = layer.forward(self.x)
        return
    
    def backward(self):
        delta = self.y_pred - self.y
        for layer in reversed(self.layers):
            delta=layer.backward(delta, self.learning_rate)

    def train(self, x, y, x_test, y_test, epochs=1000, accuracy=0.99, learning_rate=0.001):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        for _ in range(epochs):
            for i in range(len(x)):
                output = self.forward(x[i])
                delta = output - y[i]
                self.backward(delta)
                print(f"Epoch {_} - Train Accuracy: {np.mean(output.argmax(axis=1) == y[i])}", end="\r")
            print(f"Epoch {_} - Train Accuracy: {np.mean(output.argmax(axis=1) == y[i])}", end="\t")
            test_pred = self.predict(x_test).argmax(axis=1)
            print(" - Test Accuracy: {np.mean(test_pred == y_test)}")
            if np.mean(test_pred == y_test) > accuracy:
                break

    def predict(self, x):
        return np.array([self.forward(x_) for x_ in x])

    def __str__(self):
        return "\n".join([str(layer) for layer in self.layers])