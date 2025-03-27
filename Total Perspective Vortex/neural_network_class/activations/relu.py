from numpy import maximum, where

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return maximum(0, x)

    def backward(self, x):
        return where(x <= 0, 0, 1)