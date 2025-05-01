import numpy as np

class Input:
    def __init__(self, **kwargs):
        incomming_shape = kwargs.get("data_shape", None)
        if len(incomming_shape) == 1:
            self.data_shape = incomming_shape
            self.filters = None
        else:
            self.data_shape = incomming_shape[1:]
            self.filters = incomming_shape[0]

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
        return self.data_shape

    def get_output_shape(self):
        if len (self.data_shape) == 1:
            return self.data_shape[0], self.filters    
        return self.data_shape, self.filters
        
    def calculate_delta_on_input(self):
        return self.delta

    def get_delta(self):
        return None

    def forward_calculation(self, X):
        self.data_shape = X.shape
        return X

    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        return self.delta
        #return delta_lext_layer.reshape(delta_lext_layer.shape[0], self.data_shape[1], self.data_shape[2])

    def backward_calculation_last_layer(self, delta):
        return delta
    
    def __str__(self):
        return f"Input Layer{self.shape()}"
