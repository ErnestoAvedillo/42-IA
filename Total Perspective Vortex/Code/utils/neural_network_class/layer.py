from .layers.dense import Dense
from .layers.conv import Conv2D
from .layers.flattend import Flattend
from .layers.max_pool import MaxPool
from .layers.input import Input

class Layer():
    def __init__(self, layer_type = None, **kwargs):
        self.layers ={
            'dense': Dense,
            'conv': Conv2D,
            'flattend': Flattend,
            'max_pool': MaxPool,
            'input': Input
        }
        if layer_type not in self.layers:
            raise ValueError(f"Invalid layer_type: {layer_type}.")
        self.layer_type = layer_type
        #self.layer = layers[layer_type](**kwargs)
        self.layer = self.layers[layer_type](**kwargs)

    def forward_calculation(self, X):
        return self.layer.forward_calculation(X)
    
    def backward_calculation_last_layer(self, delta):
        self.layer.backward_calculation_last_layer(delta)

    def backward_calculation(self, delta):
        self.layer.backward_calculation(delta)

    def get_weights(self):
        return self.layer.get_weights()
    
    def get_bias(self):
        return self.layer.get_bias()

    def get_delta(self):
        return self.layer.get_delta()
    
    def get_model(self):
        model ={"layer_type": self.layer_type,
                "model": self.layer.get_model()}
        return model

    def calculate_delta_on_input(self):
        return self.layer.calculate_delta_on_input()
    
    def get_data_shape(self):
        return self.layer.get_data_shape()
    
    def get_output_shape(self):
        return self.layer.get_output_shape()
    
    def set_model(self, **model):
        self.layer.set_model(**model)
        return

    def set_optimizer(self, optimizer):
        return self.layer.set_optimizer(optimizer)
    
    def __str__ (self):
        return self.layer.str()