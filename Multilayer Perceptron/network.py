import numpy as np
from layer import Layer

class Network:
    def __init__(self,layers = None, x = None, y = None, learning_rate = 0.001):
        if layers is None:
            self.layers = []
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
        self.layers[-1].backward_last_layer(delta, self.learning_rate)
        for i in range(len(self.layers[:-1]) - 1, -1, -1):
            self.layers[i].backward(self.learning_rate, self.layers[i+1])

    def train(self, X, Y, X_test, y_test, epochs=1000, accuracy=0.99, learning_rate=0.01, batch_size=None):
     #   factor_theta0 = np.concat((np.ones([1]),-(X.mean(axis = 0)/ X.std(axis = 0))), axis = 0)
        m = Y.shape[0]
        if X.ndim == 1:
            X = X.reshape(m, 1)
        if Y.ndim == 1:
            Y = Y.reshape(m, 1)
        m = y_test.shape[0]
        if X.ndim == 1:
            x_test = x_test.reshape(m, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(m, 1)
        X = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
        x_test = (X_test - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
        y_pred = np.zeros(Y.shape)
        test_pred = np.zeros(y_test.shape)
        self.learning_rate = learning_rate
        for _ in range(epochs):
            if batch_size is None:
                y_pred = self.forward(X)
                delta = y_pred - Y
                self.backward(delta)
                print(f"Epoch {_} - Train Accuracy: {np.mean(np.round(y_pred).astype(int) == Y)}", end="\t")
            else:
                batches_x = np.array_split(x, np.ceil(x.shape[0] / batch_size))
                batches_y = np.array_split(y, np.ceil(y.shape[0] / batch_size))
                i = 0
                for i in range(batches_x):
                    x = batches_x[i]
                    y = self.forward(x)
                    delta = batches_y[i] - y
                    self.backward(delta)
                    print(f"Epoch {_} -batch {i} - Train Accuracy: {np.mean(np.round(y_pred).astype(int) == Y)}", end="\r")
                y_pred = self.forward(X)
                print(f"Epoch {_} - Train Accuracy: {np.mean(np.round(y_pred).astype(int) == Y)}", end="\t")
            test_pred = self.forward(x_test)
#            train_pred = self.predict(X)
#            for i in range(20):
#                print(f" - compare train: {np.round(train_pred[i],2)} == {np.round(y_test[i],2)}")
#            for i in range(20):
#                print(f" - compare  test: {np.round(test_pred[i],2)} == {np.round(y_test[i],2)}")
            print(f" - Test Accuracy: {np.mean(np.round(test_pred).astype(int) == y_test)}")
#            input("Press Enter to continue...")
            if np.mean(test_pred == y_test) > accuracy:
                break
        return np.mean(np.round(test_pred).astype(int) == y_test)
        

    def predict(self, x):
        return self.forward(x)

    def __str__(self):
        return "\n".join([str(layer) for layer in self.layers])