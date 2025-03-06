import numpy as np
optimizers = {"sgd":1,"momentum":2,"nesterov":3,"adagrad":4,"rmsprop":5,"adam":6}
class Optimizer:
    #def __init__(self, optimizer = "sgd", learning_rate = 0.01, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,**kwargs):
    def __init__(self, **kwargs):
        if "optimizer" not in kwargs:
            kwargs["optimizer"]= "sgd"
#        self.learning_rate = learning_rate
#        self.momentum = momentum
#        self.beta1 = beta1
#        self.beta2 = beta2
#        self.epsilon = epsilon  
#        self.velocities_weights = 0
#        self.velocities_bias = 0
        self.optimizer = optimizers[kwargs["optimizer"]]
        self.init_optimizer_functions = {
            1: self.init_sgd,
            2: self.init_momentum,
            3: self.init_nesterov,
            4: self.init_adagrad,
            5: self.init_rmsprop,
            6: self.init_adam
        }
        self.init_optimizer_functions[self.optimizer](**kwargs)
        self.optimizer_functions = {
            1: self.sgd_optimizer,
            2: self.momentum_optimizer,
            3: self.nesterov_optimizer,
            4: self.adagrad_optimizer,
            5: self.rmsprop_optimizer,
            6: self.adam_optimizer
        }

    def init_sgd(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        return
    
    def init_momentum(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.momentum = kwargs.get("momentum", 0.9)
        self.velocities_weights = 0
        self.velocities_bias = 0
        return
    
    def init_nesterov(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.momentum = kwargs.get("momentum", 0.9)
        self.velocities_weights = 0
        self.velocities_bias = 0
        return

    def init_adagrad(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.gradients_weights = 0
        self.gradient_bias = 0
        return

    def init_rmsprop(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.beta = kwargs.get("beta", 0.9)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.velocities_weights = None
        self.velocities_bias = None
        return
    
    def init_adam(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.index = 0
        self.momentum_weights = None
        self.momentum_bias = None
        self.velocities_weights = None
        self.velocities_bias = None
        return

    def calculate_optimizer(self,gradients_weights, gradient_bias):
        return self.optimizer_functions[self.optimizer](gradients_weights, gradient_bias)
    
    def sgd_optimizer(self, gradients_weights, gradient_bias):
        return self.learning_rate * gradients_weights,  self.learning_rate * gradient_bias
    
    def momentum_optimizer(self, gradients_weights, gradient_bias):
        self.velocities_weights = self.learning_rate * (self.momentum * self.velocities_weights + (1 - self.momentum) * gradients_weights)
        self.velocities_bias = self.learning_rate * (self.momentum * self.velocities_bias +  (1 - self.momentum) * gradient_bias)
        return self.velocities_weights, self.velocities_bias

    def nesterov_optimizer(self, gradients_weights, gradient_bias):
        self.velocities_weights = - self.momentum * self.velocities_weights + self.learning_rate * gradients_weights
        self.velocities_bias = - self.momentum * self.velocities_bias + self.learning_rate * gradient_bias
        return self.velocities_weights, self.velocities_bias

    def adagrad_optimizer(self, gradients_weights, gradient_bias):
        self.gradients_weights += gradients_weights ** 2
        self.velocities_weights = self.learning_rate * gradients_weights / np.sqrt(self.gradients_weights + self.epsilon)
        self.gradient_bias += gradient_bias ** 2
        self.velocities_weights = self.learning_rate * gradient_bias / np.sqrt(self.gradient_bias + self.epsilon)
        return self.velocities_weights, self.velocities_weights
    
    def rmsprop_optimizer(self, gradients_weights, gradient_bias):
        if self.velocities_weights is None:
            self.velocities_weights = np.zeros_like(gradients_weights)
            self.velocities_bias = np.zeros_like(gradient_bias)
        self.velocities_weights = self.beta * self.velocities_weights + (1 - self.beta) * gradients_weights ** 2
        gradients_weights_correction = self.learning_rate * gradients_weights / (np.sqrt(self.velocities_weights) + self.epsilon)
        self.velocities_bias = self.beta * self.velocities_bias + (1 - self.beta) * gradient_bias ** 2
        gradients_bias_correction = self.learning_rate * gradient_bias / (np.sqrt(self.velocities_bias) + self.epsilon)
        return gradients_weights_correction, gradients_bias_correction
    
    def adam_optimizer(self, gradients_weights, gradient_bias):
        if self.velocities_weights is None:
            self.velocities_weights = np.zeros_like(gradients_weights)
            self.velocities_bias = np.zeros_like(gradient_bias)
            self.momentum_weights = np.zeros_like(gradients_weights)
            self.momentum_bias = np.zeros_like(gradient_bias)
        self.momentum_weights = self.beta1 * self.momentum_weights + (1 - self.beta1) * gradients_weights
        self.velocities_weights = self.beta2 * self.velocities_weights + (1 - self.beta2) * gradients_weights ** 2
        
        gradients_weights_correction = self.learning_rate * self.momentum_weights / (np.sqrt(self.velocities_weights) + self.epsilon)
        return gradients_weights_correction, gradients_bias_correction