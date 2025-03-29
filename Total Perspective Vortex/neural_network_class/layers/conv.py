import numpy as np
import random
from scipy import signal
from ..activation import Activation
from ..optimizer import Optimizer


class Conv2D(Activation):
    def __init__(self, **kwargs):
        self.data_shape = kwargs.get("data_shape", None)
        self.kernel_size = kwargs.get("kernel_size", None)
        self.filters = kwargs.get("filters", 1)
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", 0)
        self.weights = np.random.randn(self.filters, self.kernel_size, self.kernel_size)
        #self.weights = np.random.randn(self.kernel_size, self.kernel_size)
        self.bias = np.random.randn(self.filters)
        #self.bias = random.random()
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
                 "data_shape": self.data_shape,
                 "kernel_size": self.kernel_size,
                 "filters": self.filters,
                 "stride": self.stride,
                 "padding": self.padding}
        return model

    def set_model(self, model):
        self.weights = np.array(model["weights"])
        self.bias = np.array(model["bias"])
        self.data_shape = model["data_shape"]
        self.kernel_size = model["kernel_size"]
        self.filters = model["filters"]
        self.stride = model["stride"]
        self.padding = model["padding"]

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def get_data_shape(self):
        return self.data_shape, self.filters
    
    def calculate_delta_on_input(self):
        output = np.zeros(self.input.shape)
        weights_rot = np.flip(self.weights)
        for i in range(self.delta.shape[0]):
            for j in range(self.delta.shape[1]):
                output[i,j] = signal.convolve2d(self.delta[i,j], weights_rot[j], mode = "full")
        return output

    def get_output_shape(self):
        #arr1 = np.ones((self.data_shape))
        #arr2 = np.ones((self.kernel_size,self.kernel_size))
        #output = signal.convolve2d(arr1, arr2, mode = "valid").shape
        output1 = self.data_shape[0] - self.kernel_size + 1
        output2 = self.data_shape[1] - self.kernel_size + 1
        return (output1, output2), self.filters

    def forward_calculation(self, X):
        self.input = X
        self.n_samples = X.shape[0]
        outputs = []
        for i in range(self.n_samples):
            outputs1 = []
            for l in range(0,self.filters):
                aux = signal.convolve2d(X[i, l, :, :], self.weights[l, :, :], mode = "valid")+ self.bias
                outputs1.append(aux)
            outputs1 = np.stack(outputs1)
            outputs.append(outputs1)
        output = np.stack(outputs)        
        output = self.forward(output)
        return output
    
    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        self.delta_bias = np.sum(self.delta, axis = (0, 2, 3))
        self.delta_inputs = np.copy(self.input)
        for i in range(self.delta.shape[0]):
            for j in range(self.delta.shape[1]):
                self.delta_weights[j]= signal.correlate2d(self.input[i, j, :, :], self.delta[i, j, :, :], "valid")
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias