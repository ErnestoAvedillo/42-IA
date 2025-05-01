import numpy as np

class Flattend:
    def __init__(self, **kwargs):
        self.data_shape = kwargs.get("data_shape", None)
        self.filters = kwargs.get("filters", 1)
        self.delta = None

    def get_model(self):
        return None
    
    def set_model(self, model):
        return
    
    def set_optimizer(self, optimizer):
        return

    def get_weights(self):
        return None

    def get_bias(self):
        return None
    
    def get_data_shape(self):
        return self.data_shape, self.filters

    def get_output_shape(self):
        aux = np.prod(self.data_shape) * self.filters
        return int(aux), None
    
    def calculate_delta_on_input(self):
        return self.delta.reshape(self.data_shape)

    def get_delta(self):
        return None

    def forward_calculation(self, X):
        self.data_shape = X.shape
        return X.reshape(self.data_shape[0], -1)

    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        return self.delta.reshape(self.data_shape)
        #return delta_lext_layer.reshape(delta_lext_layer.shape[0], self.data_shape[1], self.data_shape[2])

    def backward_calculation_last_layer(self, delta):
        return delta
    
    def __str__(self):
        return f"Flattend Layer{self.shape()}"
