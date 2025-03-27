from numpy import tanh

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        return tanh(x)

    def backward(self, x):
        return 1.0 - x ** 2