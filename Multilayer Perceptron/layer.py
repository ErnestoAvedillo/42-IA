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

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def shape(self):
        return np.array(self.input_dim,self.nodes)
    
    def forward(self, input):
        self.input = input
        self.y_predicted = self.activation.forward(np.dot(input, self.weights) + self.bias)
        return self.y_predicted

    def backward(self, delta, learning_rate):
        gradient = delta[:, np.newaxis, :] * self.activation.backward(self.input)[:, :, np.newaxis]
        #delta_weights = learning_rate * np.dot(self.input, gradient)
        delta_weights = learning_rate * np.einsum('ij,ijk -> jk', self.input, gradient)
        delta_bias = np.sum(np.sum(gradient, axis=0),axis = 0)
        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * delta_bias
        return gradient

    def __str__(self):
        return f"Layer: {self.input_dim} -> {self.nodes} Activation: {self.activation}"