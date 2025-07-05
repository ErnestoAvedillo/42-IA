import numpy as np

class Adagrad:

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.gradients_weights = 0
        self.gradient_bias = 0
        return

    def optimize(self, gradients_weights, gradient_bias):
        self.gradients_weights += gradients_weights ** 2
        self.velocities_weights = self.learning_rate * gradients_weights / np.sqrt(self.gradients_weights + self.epsilon)
        self.gradient_bias += gradient_bias ** 2
        self.velocities_weights = self.learning_rate * gradient_bias / np.sqrt(self.gradient_bias + self.epsilon)
        return self.velocities_weights, self.velocities_weights