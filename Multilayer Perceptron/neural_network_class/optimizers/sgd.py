import numpy as np

class SGD:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        return

    def optimize(self, gradients_weights, gradient_bias):
        return self.learning_rate * gradients_weights,  self.learning_rate * gradient_bias