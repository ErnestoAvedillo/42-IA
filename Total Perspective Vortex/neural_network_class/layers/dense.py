import numpy as np  
from ..activation import Activation
from ..optimizer import Optimizer

class Dense(Activation):
    def __init__(self, **kwargs):
        data_shape = kwargs.get("data_shape", None)
        input_shape = kwargs.get("input_shape", None)
        model = kwargs.get("model", None) 
        if len(kwargs) == 0:
            return
        if model is not None:
            self.set_model(model = model)
        else:
           if data_shape is None or input_shape is None:
                raise ValueError("data_shape and input_shape must be defined")
           self.data_shape = data_shape
           self.input_shape = input_shape
           Activation.__init__(self, type = kwargs.get("activation", "sigmoid"))
           self.weights = np.random.randn(data_shape, input_shape)
           self.bias = np.random.randn(input_shape)
           self.input = None
           self.y_predicted = None
           self.delta = None
           self.delta_weights = None
           self.delta_bias = None
           self.data_length = None
           self.optimizer = None
    
    def set_optimizer(self, optimizer = Optimizer(optimizer = "sgd")):
        self.optimizer = optimizer
    
    def get_model(self):
        model = {"activation":self.get_activation(),
                 "weights":self.weights.tolist(),
                 "bias": self.bias.tolist(),
                 "data_shape": self.data_shape,
                 "input_shape": self.input_shape}
        return model

    def set_model(self, model):
        self.set_activation(type = model["activation"])
        self.weights = np.array(model["weights"])
        self.bias = np.array(model["bias"])
        self.data_shape = model["data_shape"]
        self.input_shape = model["input_shape"]

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias

    def get_delta(self):
        return self.delta
    
    def calculate_delta_on_input(self):
        return np.dot(self.get_delta(), self.get_weights().T)
    
    def get_input_shape(self):
        return self.data_shape
    
    def get_output_shape(self):
        return self.input_shape
    
    def forward_calculation(self, input):
        self.input = input
        self.data_length = input.shape[0]
        self.y_predicted = self.forward(np.dot(input, self.weights) + self.bias)
        if np.isnan(self.y_predicted).any():
            print("nan")
        return self.y_predicted

    def backward_calculation_last_layer(self, error):
        activation = self.backward(np.dot(self.input, self.weights))
        self.delta = error * activation
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias
        if np.isnan(self.weights).any() or np.isnan(self.bias).any():
            print("nan")
        return 
    
    def backward_calculation(self, next_layer):
        #aux = np.dot(next_layer.get_delta(), next_layer.get_weights().T)
        aux = next_layer.calculate_delta_on_input()
        aux0 = np.dot(self.input, self.weights)
        aux1 = self.backward(aux0)
        self.delta = aux * aux1
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias
        if np.isnan(self.weights).any() or np.isnan(self.bias).any():
            print("nan")
        return

    def __str__(self):
        return f"Layer: weights {self.weights} -> bias {self.bias} -- delta: {self.delta_weights} -> {self.delta_bias}"