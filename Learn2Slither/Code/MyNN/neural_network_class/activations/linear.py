from numpy import ones, argmax
from sklearn.metrics import accuracy_score, mean_squared_error


class Linear:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, x):
        return ones(x.shape)

    def predict_output(self, x):
        return x

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        if len(Y.shape) > 1:
            Y_labels = argmax(Y, axis=1)
            y_calc_labels = argmax(y_predicted, axis=1)
            metrics = {"loss": mean_squared_error(Y, y_calculated),
                       "accuracy": accuracy_score(Y_labels, y_calc_labels)}
        else:
            metrics = {"loss": mean_squared_error(Y, y_calculated),
                       "accuracy": accuracy_score(Y, y_predicted)}
        return metrics