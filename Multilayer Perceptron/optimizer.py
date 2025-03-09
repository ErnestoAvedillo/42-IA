import copy
from optimizers.adam import Adam
from optimizers.adagrad import Adagrad
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.rmsprop import RMSProp
from optimizers.sgd import SGD

list_optimizers = {"sgd":1,"momentum":2,"nesterov":3,"adagrad":4,"rmsprop":5,"adam":6}
class Optimizer:
    def __init__(self, **kwargs):
        
        opcion = list_optimizers.get(kwargs["optimizer"], 1)

        optimizer_class = {
            1: Adam,
            2: Adagrad,
            3: Momentum,
            4: Nesterov,
            5: RMSProp,
            6: SGD
        }

        self.optimizer = optimizer_class[opcion](**kwargs)
        return

    def calculate_optimizer(self,gradients_weights, gradient_bias):
        return self.optimizer.optimize(gradients_weights, gradient_bias)
    
