import numpy as np

class Momentum:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.momentum = kwargs.get("momentum", 0.9)
        self.velocities_weights = 0
        self.velocities_bias = 0
        return
    
    def optimize(self, gradients_weights, gradient_bias):
        self.velocities_weights = self.learning_rate * (self.momentum * self.velocities_weights + (1 - self.momentum) * gradients_weights)
        self.velocities_bias = self.learning_rate * (self.momentum * self.velocities_bias +  (1 - self.momentum) * gradient_bias)
        return self.velocities_weights, self.velocities_bias