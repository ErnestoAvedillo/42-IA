from numpy import exp, power, round
from sklearn.metrics import log_loss, accuracy_score

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + exp(-x))

    def backward(self, x):
        aux = exp(x)
        aux1 = aux * power(1 + aux, -2)
        return aux1

    def predict_output(self, x):
        return round(x)

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        metrics = {"loss": log_loss(Y, y_calculated),
                   "accuracy": accuracy_score(Y, y_predicted)}
        return metrics