import numpy as np

class RMSProp:

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.beta = kwargs.get("beta", 0.9)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.velocities_weights = None
        self.velocities_bias = None
        return
    
    def optimize(self, gradients_weights, gradient_bias):
        if self.velocities_weights is None:
            self.velocities_weights = np.zeros_like(gradients_weights)
            self.velocities_bias = np.zeros_like(gradient_bias)
        self.velocities_weights = self.beta * self.velocities_weights + (1 - self.beta) * gradients_weights ** 2
        gradients_weights_correction = self.learning_rate * gradients_weights / (np.sqrt(self.velocities_weights) + self.epsilon)
        self.velocities_bias = self.beta * self.velocities_bias + (1 - self.beta) * gradient_bias ** 2
        gradients_bias_correction = self.learning_rate * gradient_bias / (np.sqrt(self.velocities_bias) + self.epsilon)
        return gradients_weights_correction, gradients_bias_correction