import numpy as np
from layer import Layer
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Network:
    def __init__(self,layers = None, x = None, y = None, learning_rate = 0.001):
        if layers is None:
            self.layers = []
        self.mean = None
        self.std = None
        self.learning_rate = None

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
        self.layers.append(Layer(input_dim = input, nodes = nodes, activation = activation))

    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self,delta):
        self.layers[-1].backward_last_layer(delta, self.learning_rate)
        for i in range(len(self.layers[:-1]) - 1, -1, -1):
            self.layers[i].backward(self.learning_rate, self.layers[i+1])

    def train(self, X, Y, X_test, y_test, epochs=1000, accuracy=0.9999, learning_rate=0.01, batch_size=None):
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
        self.mean = X.mean(axis = 0)
        self.std = X.std(axis = 0)
        X = (X - self.mean) / (self.std)
        x_test = (X_test - self.mean) / (self.std)
        y_pred = np.zeros(Y.shape)
        test_pred = np.zeros(y_test.shape)
        self.learning_rate = learning_rate
        for _ in range(epochs):
            if batch_size is None:
                y_pred = self.forward(X)
                delta = y_pred - Y
                self.backward(delta)
            else:
                batches_x = np.array_split(X, np.ceil(X.shape[0] / batch_size))
                batches_y = np.array_split(Y, np.ceil(Y.shape[0] / batch_size))
                for i in range(len(batches_x)):
                    y = self.forward(batches_x[i])
                    delta = batches_y[i] - y
                    self.backward(delta)
                    y = self.forward(batches_x[i])
                    print(f"Epoch {_} -batch {i} - Train Accuracy: {np.mean(np.round(y).astype(int) == [batches_y[i]])}", end="\r")
            y_pred = self.forward(X)
            print(f"Epoch {_} - Train Accuracy: {np.mean(np.round(y_pred).astype(int) == Y)}", end="\t")
            test_pred = self.forward(x_test)
            print(f" - Test Accuracy: {np.mean(np.round(test_pred).astype(int) == y_test)}")
            curr_accuracy = np.mean(np.round(test_pred).astype(int) == y_test) 
            if curr_accuracy > accuracy :
                break
        return np.mean(np.round(test_pred).astype(int) == y_test)
        
    def save_model(self, file_name):
        model = {}
        model["mean"] = self.mean.tolist()
        model["std"] = self.std.tolist()
        model["learning_rate"] = self.learning_rate
        model["layers"] = []
        for layer in self.layers:
            model["layers"].append(layer.__dict__)
        with open(file_name, "w") as file:
            file.write(model)

    def get_model(self):
        model_layers = []
        for layer in self.layers:
            model_layers.append(layer.get_model())

        model ={
            "mean":self.mean.tolist(),
            "std":self.std.tolist(),
            "learning_rate":self.learning_rate,
            "layers":model_layers
        }
        return model

    def set_model(self, model):
        self.layers = []
        self.mean = np.array(model["mean"])
        self.std = np.array(model["std"])
        self.learning_rate = model["learning_rate"]
        for layer in model["layers"]:
            new_layer= Layer(model = layer)
            self.layers.append(new_layer)

    def save_model(self, file_name):
        model = self.get_model()
        with open(file_name, "w") as archivo:
            json.dump(model, archivo)
    
    def open_model(self, file_name):
        with open(file_name,'r') as archivo:
            model = json.load(archivo)
        self.set_model(model)

    def predict(self, X):
        x = (X - self.mean) / self.std
        y_pred =  self.forward(x)
        return np.round(y_pred).astype(int)

    def evaluate_prediction(self, Y, y_pred):
        metrics = {
            "nyaccuracy": np.mean(np.round(y_pred).astype(int) == Y),
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred, average="weighted"),
            "recall": recall_score(Y, y_pred, average="weighted"),
            "f1_score": f1_score(Y, y_pred, average="weighted"),
        }
        return metrics
            
    def __str__(self):
        return "\n".join([str(layer) for layer in self.layers])