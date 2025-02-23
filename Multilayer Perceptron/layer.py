import numpy as np  
from activation import Activation

class Layer:
    def __init__(self, input_dim, nodes, activation="sigmoid"):
        self.input_dim = input_dim
        self.nodes = nodes
        self.activation = Activation(activation)
        self.weights = np.random.randn(input_dim, nodes)
        self.bias = np.random.randn(nodes)
        self.input = None
        self.y_predicted = None
        self.delta = None
        self.delta_weights = None
        self.delta_bias = None
        self.data_length = None

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias

    def get_delta(self):
        return self.delta
    
    def shape(self):
        return np.array(self.input_dim,self.nodes)
    
    def forward(self, input):
        self.input = input
        self.data_length = input.shape[0]
        self.y_predicted = self.activation.forward(np.dot(input, self.weights) + self.bias)
        return self.y_predicted

    def backward_last_layer(self, error, learning_rate):
        self.delta = error * self.activation.backward(np.dot(self.input, self.weights))
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        self.weights -= learning_rate * self.delta_weights / self.data_length
        self.bias -= learning_rate * self.delta_bias / self.data_length
        return 
    
    def backward(self, learning_rate, next_layer):
        self.delta = np.dot(next_layer.get_delta(), next_layer.get_weights().T) * self.activation.backward(np.dot(self.input, self.weights))
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        self.weights -= learning_rate * self.delta_weights / self.data_length
        self.bias -= learning_rate * self.delta_bias / self.data_length
        return

    def __str__(self):
        return f"Layer: weights {self.weights} -> bias {self.bias} -- delta: {self.delta_weights} -> {self.delta_bias}"