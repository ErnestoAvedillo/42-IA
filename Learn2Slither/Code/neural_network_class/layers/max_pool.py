import numpy as np
from ..activation import Activation

class MaxPool(Activation):
    def __init__(self, **kwargs):
        self.data_shape = kwargs.get("data_shape", None)
        self.filters = kwargs.get("filters", 1)
        self.kernel = kwargs.get("kernel_size", 2)
        #self.step_v = kwargs.get("step_v", 1)
        #self.step_h = kwargs.get("step_h", 1)
        Activation.__init__(self, type = kwargs.get("activation", "relu"))
        self.output_max_pos = None
        if self.data_shape is None:
            raise ValueError(f"Datashape must be included in parameters.")
        self.shape_h = self.data_shape[0] // self.kernel + int(self.data_shape[0] % self.kernel > 0)
        self.shape_v = self.data_shape[1] // self.kernel + int(self.data_shape[1] % self.kernel > 0)
        self.delta = None

    def get_model(self):
        model = {
            "kernel" : self.kernel,
            "data_shape": self.data_shape,
            "filters": self.filters,
            "shape_h": self.shape_h,
            "shape_v": self.shape_v,
        #    "step_v" : self.step_v,
        #    "step_h" : self.step_h
        }
        return model
    
    def set_model(self, **kwargs):
        keys = ["kernel", "data_shape", "filters", "shape_h", "shape_v"]
        for key in keys:
            if key not in kwargs:
                raise ValueError(f"{key} is required")
        self.kernel = kwargs.get("kernel", 2)
        self.data_shape = kwargs.get("data_shape", None)
        self.filters = kwargs.get("filters", 1)
        self.shape_h = kwargs.get("shape_h", self.data_shape[0] // self.kernel + int(self.data_shape[0] % self.kernel > 0))
        self.shape_v = kwargs.get("shape_v", self.data_shape[1] // self.kernel + int(self.data_shape[1] % self.kernel > 0))
        #self.step_v = kwargs.get("step_v", 1)
        #self.step_h = kwargs.get("step_h", 1)
        return
    
    def set_optimizer(self, optimizer):
        pass

    def get_weights(self):
        pass

    def get_bias(self):
        pass
    
    def get_data_shape(self):
        return self.data_shape, self.filters

    def get_output_shape(self):
        return (self.shape_h,self.shape_v), self.filters
    
    def calculate_delta_on_input(self):
        delta_in_input = self.output_max_pos * self.delta[:, :, :, np.newaxis, :, np.newaxis]
        delta_in_input = delta_in_input.reshape(delta_in_input.shape[0], delta_in_input.shape[1], delta_in_input.shape[2] * delta_in_input.shape[3], delta_in_input.shape[4] * delta_in_input.shape[5])
        delta_in_input = delta_in_input[:, :, :self.data_shape[0], :self.data_shape[1]]
        return delta_in_input

#    def calculate_delta_on_input(self):
#        delta_input = np.zeros(self.data_shape)
#        for i in range(0, self.delta.shape[0]):
#            for j in range(0, self.delta.shape[1]):
#                for k in range(0, self.delta.shape[2], self.kernel):
#                    for l in range (0, self.delta.shape[3], self.kernel):
#                        delta_input[i,j,self.output_max_pos[i,j,k,l,0], self.output_max_pos[i,j,k,l,0]] = self.delta[i, j, k, l]
#        return delta_input

    def get_delta(self):
        return None

    def forward_calculation(self, X):
        extradims_h = 0 if  X.shape[2] % self.kernel == 0 else self.kernel - (X.shape[2] % self.kernel)
        extradims_v = 0 if  X.shape[3] % self.kernel == 0 else self.kernel - (X.shape[3] % self.kernel)
        X_copy = np.zeros((X.shape[0], self.filters, X.shape[2] + extradims_h, X.shape[3] + extradims_v))
        X_copy[:, :, :X.shape[2], :X.shape[3]] = X
        output = np.zeros((X.shape[0], self.filters, self.shape_h, self.shape_v))
        X_copy = X_copy.reshape(X_copy.shape[0],X_copy.shape[1], self.shape_h, self.kernel, self.shape_v, self.kernel)
        output = np.max(X_copy, axis = (3,5))
        output = self.forward(output)
        max_values = np.max(X_copy, axis = (3,5),keepdims = True)
        self.output_max_pos = np.equal(X_copy, max_values).astype(int)
        return output

#    def forward_calculation(self, X):
#        output = np.zeros((X.shape[0], self.filters, self.shape_h, self.shape_v))
#        self.output_max_pos = np.zeros((X.shape[0], self.filters, self.shape_h, self.shape_v, 2))
#        for i in range(0, X.shape[0]):
#            for j in range(0, self.filters):
#                for k in range(0, self.shape_h):
#                    for l in range (0,  self.shape_v):
#                        window = X[i,j,k:min(k * self.kernel + self.kernel, X.shape[2]), l:min(l * self.kernel + self.kernel, X.shape[3])]
#                        max_val = np.max(window)
#                        m,n = np.unravel_index(np.argmax(window), (self.kernel,self.kernel))
#                        absolut_max_pos = np.array([min(k * self.kernel + m,X.shape[2]),min(l * self.kernel + n, X.shape[3])])
#                        #max_value_pos = np.unravel_index(np.argmax(X[i,j, k:min(k+self.kernel,X.shape[1]), l:min(l+self.kernel,X.shape[2])]), self.kernel)
#                        output[i,j,k,l] = max_val
#                        self.output_max_pos[i,j,k,l] = absolut_max_pos # np.array([n,m])
#        output = self.forward(output)
#        return output

    def backward_calculation(self, next_layer):
        self.delta = next_layer.calculate_delta_on_input()
        return 
        #return delta_lext_layer.reshape(delta_lext_layer.shape[0], self.data_shape[1], self.data_shape[2])

    def backward_calculation_last_layer(self, delta):
        pass
    
    def __str__(self):
        return f"Flattend Layer{self.shape()}"
