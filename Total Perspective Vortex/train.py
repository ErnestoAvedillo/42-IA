
import pandas as pd
import numpy as np
from neural_network_class.network import Network
from neural_network_class.layer import Layer
from neural_network_class.optimizer import Optimizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv").values
Y = np.array([data[:, -1] == option for option in range(int(max(data[:,-1])) + 1)]).T.astype(int)
X = data[:, :-1]
X = X.reshape(-1, 64, 64)
print (f"Imprimo el shape de X {X.shape}")
print (f"Imprimo el shape de Y {Y.shape}")
print (f"Imprimo Y {Y}")
print (f"Imprimo X {X}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
network = None
network = Network()

#network.add_layer(layer = Layer(layer_type = "dense", input_shape = 64, data_shape=X_train.shape[1]))
network.add_layer(layer = Layer(layer_type = "conv", input_shape = (64,64), kernel_size = 8 , filters = 1, activation = "relu"))
#network.add_layer(layer_type = "maxpool", pool_size = (2, 2))
network.add_layer(layer_type = "flattend")
network.add_layer(layer_type = "dense", input_shape = 32)
network.add_layer(layer_type = "dense", input_shape = 16)
network.add_layer(layer_type = "dense", input_shape = 5, activation = "softmax")
#optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
#optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.001, momentum = 0.9)
#optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
#optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.4)
#optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.01, beta = 0.4, epsilon = 1e-8)
optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8)
arr_looses, arr_accuracies, arr_looses_val, arr_accuracies_val =network.train(X_train, Y_train, X_test, Y_test, epochs=1000,  optimizer = optimizer, verbose = True)

data = pd.read_csv("test.csv").values
Y = np.array([data[:, -1] == option for option in range(int(max(data[:,-1])) + 1)]).T.astype(int)
X = data[:, :-1]
X = X.reshape(-1, 64, 64)

Y_pred = network.test(X_test)
network.evaluate(Y_pred, Y_test)
for i in range(len(Y_pred)):
    print(f"Expected: {Y_test[i]} Predicted: {Y_pred[i]}")
network.save_model("model.json")

