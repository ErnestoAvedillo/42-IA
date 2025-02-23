import numpy as np
from layer import Layer

class Network:
    def __init__(self,layers = [], x = None, y = None, learning_rate = 0.001):
        self.layers = layers
        self.learning_rate = None

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_layer(self, layer = None, nodes = None, input_dim=None, activation="sigmoid"):
        if layer is not None:
            self.layers.append(layer)
            return
        if nodes is None:
            raise ValueError("nodes parameter must be defined")
        if input_dim is None and len(self.layers) == 0:
            raise ValueError("First layer must have input_dim")
        else:
            input = input_dim if input_dim is not None else self.layers[-1].nodes
        self.layers.append(Layer(input, nodes, activation))

    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self,delta):
        for layer in reversed(self.layers):
            delta=layer.backward(delta, self.learning_rate).mean(axis=2)

    def train(self, X, Y, x_test, y_test, epochs=1000, accuracy=0.99, learning_rate=0.001, batch_size=None):
        factor_theta0 = np.concat((np.ones([1]),-(X.mean(axis = 0)/ X.std(axis = 0))), axis = 0)
        X = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
        y_pred = np.zeros(Y.shape)
        self.learning_rate = learning_rate
        for _ in range(epochs):
            if batch_size is None:
                y_pred = self.forward(X)
                delta = y_pred - Y
                self.backward(delta)
                print(f"Epoch {_} - Train Accuracy: {np.mean(y_pred.argmax() == Y)}", end="\t")
            else:
                batches_x = np.array_split(x, np.ceil(x.shape[0] / batch_size))
                batches_y = np.array_split(y, np.ceil(y.shape[0] / batch_size))
                i = 0
                for i in range(batches_x):
                    x = batches_x[i]
                    y = self.forward(x)
                    delta = batches_y[i] - y
                    self.backward(delta)
                    print(f"Epoch {_} -batch {i} - Train Accuracy: {np.mean(y.argmax() == y)}", end="\r")
                y_pred = self.forward(X)
                print(f"Epoch {_} - Train Accuracy: {np.mean(y_pred.argmax(axis=1) == X)}", end="\t")
            test_pred = self.predict(x_test).argmax()
            print(f" - Test Accuracy: {np.mean(test_pred == y_test)}")
            if np.mean(test_pred == y_test) > accuracy:
                break

    def predict(self, x):
        return self.forward(x)

    def __str__(self):
        return "\n".join([str(layer) for layer in self.layers])