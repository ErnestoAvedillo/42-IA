from numpy import exp, ones, zeros_like, arange
from sklearn.metrics import log_loss, accuracy_score

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        aux = exp(x)
        return aux / aux.sum(axis=1, keepdims=True)

    def backward(self, x):
        return ones(x.shape)

    def predict_output(self, x):
        output = zeros_like(x)
        max_indices = x.argmax(axis=1)
        output[arange(x.shape[0]), max_indices] = 1
        return output

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        if Y.dtype != 'int64' or Y.dtype != 'int32':
            Y_converted = zeros_like(Y)
            max_indices = Y.argmax(axis=1)
            Y_converted[arange(Y.shape[0]), max_indices] = 1
            metrics = {"loss": log_loss(Y_converted, y_calculated),
                       "accuracy": accuracy_score(Y_converted, y_predicted)}
        else:
            metrics = {"loss": log_loss(Y, y_calculated),
                       "accuracy": accuracy_score(Y, y_predicted)}
        return metrics