from numpy import tanh, zeros_like
from sklearn.metrics import accuracy_score, mean_squared_error


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        return tanh(x)

    def backward(self, x):
        return 1.0 - x ** 2

    def predict_output(self, x):
        output = zeros_like(x)
        output[x >= 0] = 1
        output[x < 0] = 0
        return output

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        metrics = {"loss": mean_squared_error(Y, y_calculated),  # Tanh does not have a specific loss function
                   "accuracy": accuracy_score(Y, y_predicted)}
        return metrics
