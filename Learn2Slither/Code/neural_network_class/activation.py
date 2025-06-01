from .activations.sigmoid import Sigmoid
from .activations.tanh import Tanh
from .activations.relu import ReLU
from .activations.linear import Linear
from .activations.softmax import Softmax


class Activation:
    def __init__(self, type="sigmoid"):
        self.activation_class = {
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "relu": ReLU,
            "linear": Linear,
            "softmax": Softmax
        }
        if type not in self.activation_class:
            raise ValueError("Activation type not supported")
        self.type = type
        self.activation = self.activation_class[self.type]()
        pass

    def get_activation(self):
        return self.type

    def set_activation(self, type):
        self.type = type
        self.activation = self.activation_class[self.type]()

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, x):
        return self.activation.backward(x)

    def predict_output(self, x):
        return self.activation.predict_output(x)

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        return self.activation.evaluate_prediction(Y, y_calculated, y_predicted)
