from numpy import exp, tanh, maximum, where


activation_types = {"sigmoid":1, "tanh":2, "relu":3, "linear":4}
   

class Activation:
    def __init__(self, type = "sigmoid"):

        if type not in activation_types:
            raise ValueError("Activation type not supported")
        self.type = activation_types[type]
        self.forward_functions = {1: self.sigmoid, 2: self.tanh, 3: self.relu, 4: lambda x: x}
        self.backward_functions = {1: self.sigmoid_derivative, 2: self.tanh_derivative, 3: self.relu_derivative, 4: lambda x: 1} # linear derivative is 1, so we can just return x and it will
        pass

    def forward(self, x):
        return self.forward_functions[self.type](x)

    def backward(self, x):
        return self.backward_functions[self.type](x)

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