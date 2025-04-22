import numpy as np
from sgd import SGD
from momentum import Momentum
from nesterov import Nesterov
from adagrad import Adagrad
from rmsprop import RMSProp
from adam import Adam

optimizers = {"sgd":1,"momentum":2,"nesterov":3,"adagrad":4,"rmsprop":5,"adam":6}
class Optimizer:
    def __init__(self, **kwargs):
        if "optimizer" not in kwargs:
            kwargs["optimizer"]= "sgd"
        self.name_optimizer = optimizers[kwargs["optimizer"]]
        
        self.classes_optimizers = {
            1: self.init_sgd,
            2: self.init_momentum,
            3: self.init_nesterov,
            4: self.init_adagrad,
            5: self.init_rmsprop,
            6: self.init_adam
        }
        self.init_optimizer_functions[self.optimizer](**kwargs)
        return
        
    def init_sgd(self, **kwargs):
        self.optimizer = SGD(**kwargs)
        return
    
    def init_momentum(self, **kwargs):
        self.optimizer = Momentum(**kwargs)
        return
    
    def init_nesterov(self, **kwargs):
        self.optimizer = Nesterov(**kwargs)
        return

    def init_adagrad(self, **kwargs):
        self.optimizer = Adagrad(**kwargs)
        return
    
    def init_rmsprop(self, **kwargs):
        self.optimizer = RMSProp(**kwargs)
        return
    
    def init_adam(self, **kwargs):
        self.optimizer = Adam(**kwargs)
        return
    
    def calculate_optimizer(self,gradients_weights, gradient_bias):
        return self.optimizer.optimize(gradients_weights, gradient_bias)

    
