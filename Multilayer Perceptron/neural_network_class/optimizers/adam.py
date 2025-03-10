import numpy as np

class Adam:

    def __init__(self, **kwargs):
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
    
    def optimize(self, gradients_weights, gradient_bias):
        if self.velocities_weights is None:
            self.velocities_weights = np.zeros_like(gradients_weights)
            self.velocities_bias = np.zeros_like(gradient_bias)
            self.momentum_weights = np.zeros_like(gradients_weights)
            self.momentum_bias = np.zeros_like(gradient_bias)
        self.momentum_weights = self.beta1 * self.momentum_weights + (1 - self.beta1) * gradients_weights
        self.velocities_weights = self.beta2 * self.velocities_weights + (1 - self.beta2) * gradients_weights ** 2
        gradients_weights_correction = self.learning_rate * self.momentum_weights / np.sqrt(self.velocities_weights + self.epsilon)
        self.momentum_bias = self.beta1 * self.momentum_bias + (1 - self.beta1) * gradient_bias
        self.velocities_bias = self.beta2 * self.velocities_bias + (1 - self.beta2) * gradient_bias ** 2
        gradients_bias_correction = self.learning_rate * self.momentum_bias / np.sqrt(self.velocities_bias + self.epsilon)
        return gradients_weights_correction, gradients_bias_correction