from numpy import exp, ones
class Linear:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, x):
        return ones(x.shape)