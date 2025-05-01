from numpy import exp, ones

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        aux = exp(x)
        return aux / aux.sum(axis = 1,keepdims=True)

    def backward(self, x):
        return ones(x.shape)