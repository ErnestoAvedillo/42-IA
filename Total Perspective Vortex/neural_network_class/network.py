import numpy as np
from .layers.dense import Dense
from .layer import Layer
from .optimizer import Optimizer
import json
import copy


class Network:
    #def __init__(self,layers = None, x = None, y = None, learning_rate = 0.001):
    def __init__(self,layers = None, normalize = False):
        if layers is None:
            self.layers = []
        self.normalize = normalize
        self.mean = None
        self.std = None
        self.learning_rate = None
        self.metrics = {}
        self.metrics_val = {}

    #def add_layer(self, layer = None, layer_type = None, input_shape = None, data_shape=None, activation="sigmoid"):
    def add_layer(self, layer = None, layer_type = None, **kwargs):
        if layer is not None:
            self.layers.append(layer)
            return
        if layer_type is None:
            raise ValueError("layer_type parameter must be defined")
        if kwargs.get("data_shape",None) is None and len(self.layers) == 0:
            raise ValueError("First layer must have data_shape defined")
        elif kwargs.get("data_shape",None) is None:
            mylast_layer = self.layers[-1]
            kwargs["data_shape"],kwargs["filters"] = mylast_layer.get_output_shape()
        #self.layers.append(Layer(layer_type = layer_type ,data_shape = input, input_shape = data_shape, activation = activation))
        self.layers.append(Layer(layer_type = layer_type, **kwargs))

    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward_calculation(current_input)
        return current_input
    
    def backward(self,delta):
        self.layers[-1].backward_calculation_last_layer(delta)
        for i in range(len(self.layers[:-1]) - 1, -1, -1):
            self.layers[i].backward_calculation(self.layers[i+1])

    def train(self, X, Y, X_test, y_test, epochs=1000, delta_accuracy=0.000001, batch_size=None, optimizer=Optimizer(optimizer = "sgd"), verbose = False):
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
        if self.normalize:
            self.mean = X.mean(axis = 0)
            self.std = X.std(axis = 0)
            X = (X - self.mean) / (self.std)
            x_test = (X_test - self.mean) / (self.std)
        else:
            self.mean = 0
            self.std = 1
            x_test = X_test
        y_pred = np.zeros(Y.shape)
        test_pred = np.zeros(y_test.shape)
        for layer in  self.layers:
            optimizer_copy = copy.deepcopy(optimizer)
            layer.set_optimizer(optimizer_copy)
        array_looses = []
        array_accuracies = []
        array_looses_val = []
        array_accuracies_val = []
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
                    self.metrics = self.evaluate_prediction(Y, y_pred)
                    if verbose:
                        print(f"Epoch {_} -batch {i} - Train loss: {self.metrics['loss']:.4f}", end="\r")
            y_pred = self.forward(X)
            self.metrics = self.evaluate_prediction(Y, y_pred)
            if verbose:
                print(f"Epoch {_} - Train loss: { self.metrics['loss']:.4f}", end="\t")
            test_pred = self.forward(x_test)
            self.metrics_val = self.evaluate_prediction(y_test, test_pred)
            if verbose:
                print(f" - Val loss: {self.metrics_val['loss']:.4f} - Val accuracy: { self.metrics_val['accuracy']:.4f}")
            array_accuracies.append(self.metrics['accuracy'])
            array_looses.append(self.metrics['loss'])
            array_accuracies_val.append(self.metrics_val['accuracy'])
            array_looses_val.append(self.metrics_val['loss'])
            if len(array_accuracies) > int(0.1* epochs) and abs(self.metrics['accuracy'] - array_accuracies[-min(50,len(array_accuracies))]) < delta_accuracy :
                break
        if not verbose:
             print(f"Val loss: {self.metrics['loss']} - Val accuracy: { self.metrics['accuracy']:.4f}")
        return np.array(array_looses), np.array(array_accuracies), np.array(array_looses_val), np.array(array_accuracies_val)
        
    def save_model(self, file_name):
        model = {}
        model["mean"] = self.mean.tolist() if self.normalize else 0
        model["std"] = self.std.tolist() if self.normalize else 1
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
            "mean":self.mean.tolist() if self.normalize else 0,
            "std":self.std.tolist() if self.normalize else 1,
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
            new_layer= Dense(model = layer)
            self.layers.append(new_layer)

    def save_model(self, file_name):
        model = self.get_model()
        with open(file_name, "w") as archivo:
            json.dump(model, archivo)
    
    def open_model(self, file_name):
        with open(file_name,'r') as archivo:
            model = json.load(archivo)
        self.set_model(model)

    def predict(self, X, Y = None):
        x = (X - self.mean) / self.std
        y_pred =  self.forward(x)
        if Y is not None:
            self.metrics = self.evaluate_prediction(Y, y_pred)
        return np.round(y_pred).astype(int)

    def evaluate_prediction(self, Y, y_pred):
        y_pred = y_pred +1e-10
        y_pred_rounded = np.round(y_pred)
        not_Y = np.logical_not(Y).astype(int)
        not_y_pred_rounded = np.logical_not(y_pred_rounded).astype(int)
        metrics = {
            "loss": -np.mean(np.sum(Y * np.log(y_pred + 1e-9), axis=1)),
            "accuracy": np.mean(np.round(y_pred).astype(int) == Y),
            "f_1 score": 2 * np.sum(Y * y_pred_rounded) /(2 *np.sum(Y * y_pred_rounded) + np.sum(not_Y * y_pred_rounded) + np.sum(Y * not_y_pred_rounded) + +1e-10),
            "precision": np.sum(Y * y_pred_rounded) / (np.sum(Y * y_pred_rounded) + np.sum(not_Y * y_pred_rounded) +1e-10),
            "recall": np.sum(Y * y_pred_rounded) / (np.sum(Y * y_pred_rounded) + np.sum( Y * not_y_pred_rounded) +1e-10)
        }
        return metrics
            
    def __str__(self):
        description = f"NN with optimizer{str(self.optimizer)}.\n"
        return description.join([str(layer) for layer in self.layers])