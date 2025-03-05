import numpy as np
optimizers = {"sgd":1,"momentum":2,"nesterov":3,"adagrad":4,"rmsprop":5,"adam":6}
class Optimizer:
    def __init__(self, optimizer = "sgd", lerning_rate = 0.01, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.learning_rate = lerning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon  
        self.optimizer = optimizers[optimizer]
        self.velocities = 0
        optimizer_functions = {
            1: self.sgd_optimizer,
            2: self.momentum_optimizer,
            3: self.nesterov_optimizer,
            4: self.adagrad_optimizer,
            5: self.rmsprop_optimizer,
            6: self.adam_optimizer
        }
    def sgd_optimizer(self, gradients):
        return self.learning_rate * gradients
    
    def momentum_optimizer(self, gradients):
        self.velocities = self.momentum * self.velocities + self.learning_rate * gradients
        return self.velocities

    def nesterov_optimizer(self, gradients):
        self.velocities = self.momentum * self.velocities + self.learning_rate * gradients
        return self.momentum * self.velocities + self.learning_rate * gradients

    def adagrad_optimizer(self, gradients):
        self.velocities += gradients ** 2
        return self.learning_rate * gradients / (np.sqrt(self.velocities) + self.epsilon)
    
    def rmsprop_optimizer(self, gradients):
        self.velocities = self.beta1 * self.velocities + (1 - self.beta1) * gradients ** 2
        return self.learning_rate * gradients / (np.sqrt(self.velocities) + self.epsilon)
    
    def adam_optimizer(self, gradients):    
        self.velocities = self.beta1 * self.velocities + (1 - self.beta1) * gradients
        self.velocities = self.velocities / (1 - self.beta1)
        self.velocities = self.beta2 * self.velocities + (1 - self.beta2) * gradients ** 2
        self.velocities = self.velocities / (1 - self.beta2)
        return self.learning_rate * gradients / (np.sqrt(self.velocities) + self.epsilon)