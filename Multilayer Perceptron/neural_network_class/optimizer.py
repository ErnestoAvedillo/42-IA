import copy
from optimizers.adam import Adam
from optimizers.adagrad import Adagrad
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.rmsprop import RMSProp
from optimizers.sgd import SGD

class Optimizer:
    def __init__(self, **kwargs):
        
        opcion = kwargs.pop("optimizer", "sgd")

        optimizer_class = {
            "sgd": SGD,
            "momentum": Momentum,
            "nesterov": Nesterov,
            "adagrad": Adagrad,
            "rmsprop": RMSProp,
            "adam": Adam
        }

        self.optimizer = optimizer_class[opcion](**kwargs)
        return

    def calculate_optimizer(self,gradients_weights, gradient_bias):
        return self.optimizer.optimize(gradients_weights, gradient_bias)
    
