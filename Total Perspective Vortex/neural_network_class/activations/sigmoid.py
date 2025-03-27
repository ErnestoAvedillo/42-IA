from numpy import exp, power

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + exp(-x))

    def backward(self, x):
        aux = exp(x)
        aux1 = aux * power(1 + aux, -2)
        return aux1