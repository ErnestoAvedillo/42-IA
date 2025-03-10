from .activations.sigmoid import Sigmoid
from .activations.tanh import Tanh
from .activations.relu import ReLU
from .activations.linear import Linear
from .activations.softmax import Softmax


class Activation:
    def __init__(self, type = "sigmoid"):
        activation_class = {
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "relu": ReLU,
            "linear": Linear,
            "softmax": Softmax
        }
        if type not in activation_class:
            raise ValueError("Activation type not supported")
        self.type = type
        self.activation = activation_class[self.type]()
        pass

    def get_activation(self):
        return self.type

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, x):
        return self.activation.backward(x)
