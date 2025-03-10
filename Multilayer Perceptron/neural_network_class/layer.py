import numpy as np  
from activation import Activation
from optimizer import Optimizer

class Layer:
    #def __init__(self, input_dim = None, nodes = None, activation="sigmoid", model = None):
    def __init__(self, **kwargs):
        input_dim = kwargs.get("input_dim", None)
        nodes = kwargs.get("nodes", None)
        model = kwargs.get("model", None) 
        if len(kwargs) == 0:
            return
        if model is not None:
            self.set_model(model = model)
        else:
           if input_dim is None or nodes is None:
                raise ValueError("input_dim and nodes must be defined")
           self.input_dim = input_dim
           self.nodes = nodes
           self.activation = Activation(kwargs.get("activation", "sigmoid"))
           self.weights = np.random.randn(input_dim, nodes)
           self.bias = np.random.randn(nodes)
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
        model = {"activation":self.activation.get_activation(),
                 "weights":self.weights.tolist(),
                 "bias": self.bias.tolist(),
                 "input_dim": self.input_dim,
                 "nodes": self.nodes}
        return model

    def set_model(self, model):
        self.activation = Activation(model["activation"])
        self.weights = np.array(model["weights"])
        self.bias = np.array(model["bias"])
        self.input_dim = model["input_dim"]
        self.nodes = model["nodes"]

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

    def backward_last_layer(self, error):
        activation = self.activation.backward(np.dot(self.input, self.weights)) 
        self.delta = error * activation
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias
        return 
    
    def backward(self, next_layer):
        self.delta = np.dot(next_layer.get_delta(), next_layer.get_weights().T) * self.activation.backward(np.dot(self.input, self.weights))
        self.delta_weights = np.einsum('ij,ik->ikj', self.delta, self.input).sum(axis=0)
        self.delta_bias = np.sum(self.delta, axis=0)
        velocity_weight, velocity_bias = self.optimizer.calculate_optimizer(self.delta_weights, self.delta_bias)
        self.weights -= velocity_weight
        self.bias -=  velocity_bias
        return

    def __str__(self):
        return f"Layer: weights {self.weights} -> bias {self.bias} -- delta: {self.delta_weights} -> {self.delta_bias}"