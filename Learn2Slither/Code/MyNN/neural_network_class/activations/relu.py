from numpy import maximum, where


class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return maximum(0, x)

    def backward(self, x):
        return where(x <= 0, 0, 1)

    def predict_output(self, x):
        return maximum(0, x)

    def evaluate_prediction(self, Y, y_calculated, y_predicted):
        metrics = {"loss": None,  # ReLU does not have a loss function
                   "accuracy": None}
        return metrics