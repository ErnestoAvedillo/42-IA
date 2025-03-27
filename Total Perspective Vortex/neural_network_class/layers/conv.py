import numpy as np
import random
from scipy import signal
from ..activation import Activation
from ..optimizer import Optimizer


class Conv2D(Activation):
    def __init__(self, **kwargs):
        self.input_shape = kwargs.get("input_shape", None)
        self.kernel_size = kwargs.get("kernel_size", None)
        #self.n_filters = kwargs.get("n_filters", 1)
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", 0)
        #self.weights = np.random.randn(self.n_filters, self.kernel_size, self.kernel_size)
        self.weights = np.random.randn(self.kernel_size, self.kernel_size)
        #self.bias = np.random.randn(self.n_filters)
        self.bias = random.random()
        self.input = None
        self.delta = None
        self.delta_input = None
        self.delta_weights = np.copy(self.weights)
        self.delta_bias = np.copy(self.bias)
        self.optimizer = None
        self.n_samples = None
        Activation.__init__(self, type = kwargs.get("activation", "sigmoid"))

    def set_optimizer(self, optimizer = Optimizer(optimizer = "sgd")):
        self.optimizer = optimizer

    def get_model(self):
        model = {"weights":self.weights.tolist(),
                 "bias": self.bias.tolist(),
                 "input_shape": self.input_shape,
                 "kernel_size": self.kernel_size,
                 #"n_filters": self.n_filters,
                 "stride": self.stride,
                 "padding": self.padding}
        return model

    def set_model(self, model):
        self.weights = np.array(model["weights"])
        self.bias = np.array(model["bias"])
        self.input_shape = model["input_shape"]
        self.kernel_size = model["kernel_size"]
        #self.n_filters = model["n_filters"]
        self.stride = model["stride"]
        self.padding = model["padding"]

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def get_input_shape(self):
        return self.input_shape
    
    def calculate_delta_on_input(self):
        weights_rot = np.flip(self.weights)
        output = signal.convolve(self.delta, weights_rot, mode = "full")
        return output

    def get_output_shape(self):
        #arr1 = np.ones((self.input_shape))
        #arr2 = np.ones((self.kernel_size,self.kernel_size))
        #output = signal.convolve2d(arr1, arr2, mode = "valid").shape
        output2 = self.input_shape[0] - self.kernel_size + 1
        return (output2, output2)

    def forward_calculation(self, X):
        self.input = X
        self.n_samples = X.shape[0]
        output = signal.convolve2d(X[0], self.weights, mode = "valid")+ self.bias
        for i in range(1, self.n_samples):
                aux = signal.convolve2d(X[i], self.weights, mode = "valid")+ self.bias
                if i == 1:
                    output = np.stack((output, aux))
                else:
                    output = np.concatenate((output, aux.reshape(-1,aux.shape[0],aux.shape[1])),axis = 0)
        output = self.forward(output)
        return output
    
    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        self.delta_bias = np.sum(self.delta, axis = (0, 1, 2))
        self.delta_inputs = np.copy(self.input)
        for i in range(self.delta.shape[0]):
            self.delta_weights= signal.correlate2d(self.input[i], self.delta[i], "valid")
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias