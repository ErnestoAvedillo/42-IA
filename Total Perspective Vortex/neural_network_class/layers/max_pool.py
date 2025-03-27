import numpy as np
from ..activation import Activation

class MaxPool(Activation):
    def __init__(self, **kwargs):
        self.kernel = kwargs.get("kernel_size", 2)
        self.step_v = kwargs.get("step_v", 1)
        self.step_v = kwargs.get("step_h", 1)
        Activation.__init__(self, type = kwargs.get("activation", "relu"))
        self.data_shape = None
        self.output_max_pos = None
        self.delta = None

    def get_model(self):
        model = {
            "kernel" : self.kernel,
            "step_v" : self.step_v,
            "step_h" : self.step_h
        }
        return model
    
    def set_model(self, **kwargs):
        self.kernel = kwargs.get("kernel", 2)
        self.step_v = kwargs.get("step_v", 1)
        self.step_v = kwargs.get("step_h", 1)
        return
    
    def set_optimizer(self, optimizer):
        pass

    def get_weights(self):
        pass

    def get_bias(self):
        pass
    
    def get_input_shape(self):
        return self.data_shape

    def get_output_shape(self):
        pass
    
    def calculate_delta_on_input(self):
        delta_input = np.zeros(self.data_shape)
        for i in range(0, self.delta.shape[0]):
            for j in range(0, self.delta.shape[1], self.step_h):
                for k in range(0, self.delta.shape[2], self.step_h):
                    for l in range (0, self.delta.shape[3]):
                        delta_input[i,self.output_max_pos[i,j,k,l,0], self.output_max_pos[i,j,k,l,0],l] = self.delta[i, j, k, l]
        return delta_input

    def get_delta(self):
        return None

    def forward_calculation(self, X):
        output = np.zeros(X.shape[0], X.shape[1] // self.step_h + 1, X.shape[1] // self.step_v + 1, X.shape[3])
        self.output_max_pos = np.zeros(X.shape[0], X.shape[1] // self.step_h + 1, X.shape[1] // self.step_v + 1, X.shape[3],2)
        m = 0
        n = 0
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1], self.step_h):
                for k in range(0, X.shape[2], self.step_h):
                    for l in range (0, X.shape[3]):
                        window = X[i,j:min(j+self.kernel,X.shape[1]), k:min(k+self.kernel,X.shape[2]),l]
                        max_val = np.max(window)
                        output[i,m, n,l] = self.delta[i//self.kernel, j//self.kernel]
                        self.output_max_pos[i,m, n,l,:] = np.unravel_index( X[i,j:min(j+self.kernel,X.shape[1]), k:min(k+self.kernel,X.shape[2]),l], self.kernel)
                        m += 1
                    n += 1
        output = self.activation(output)
        return output

    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        return 
        #return delta_lext_layer.reshape(delta_lext_layer.shape[0], self.input_shape[1], self.input_shape[2])

    def backward_calculation_last_layer(self, delta):
        pass
    
    def __str__(self):
        return f"Flattend Layer{self.shape()}"