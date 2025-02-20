from numpy import exp, tanh, maximum, where

class Activation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return tanh(x)

    def tanh_derivative(x):
        return 1.0 - x ** 2
    
    def relu(self, x):
        return maximum(0, x)
    
    def relu_derivative(self, x):
        return where(x <= 0, 0, 1)