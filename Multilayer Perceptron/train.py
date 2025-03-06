import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from optimizer import Optimizer
from network import Network
import sys
from load_data import load_data

if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1],header = 0)
except FileNotFoundError:
    print("File not found")
    sys.exit(0)
if len(sys.argv) == 3:
    model_file = sys.argv[2]
else:
    model_file = "model.json"


#get data and split it
Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=["1_B","1_M"]).to_numpy().astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
"""
for i in range(X_train.shape[1]):
    print("Column", i)
    print(f"Mean    train: {X_train[:,i].mean()} Test: {X_test[:,i].mean()}")
    print(f"Std_dev train: {X_train[:,i].std()} Test: {X_test[:,i].std()}")
input("Press enter...")
"""

network = {}
i = 0
last_accuracy = 0

#optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.01, momentum = 0.9)
#optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
#optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.9)
optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.001, beta = 0.9, epsilon = 1e-8)
for k in range(5 , 10):
    for j in range(10,20):
        print (f"Training with {j} and {k} nodes in layers 2 and 3")
        network[i] = Network()
        network[i].add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
        network[i].add_layer(nodes = j)
        network[i].add_layer(nodes = k)
        #network[i].add_layer(nodes = 1)
        network[i].add_layer(nodes = 2, activation = "softmax")
        curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100,  optimizer = optimizer)
        #curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=32, optimizer = optimizer)
        if curr_accuracy > last_accuracy:
            print (f"Accuracy improved from {last_accuracy} to {curr_accuracy} with {j} and {k} nodes in layers 2 and 3")
            last_accuracy = curr_accuracy   
            first_layer = j
            second_layer = k
        i += 1
print(f"Best accuracy was {last_accuracy} with {first_layer} and {second_layer} nodes in layers 2 and 3")
network = None
network = Network()
network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
network.add_layer(nodes = first_layer)
network.add_layer(nodes = second_layer)
network.add_layer(nodes = 2, activation = "softmax")
network.train(X_train, Y_train, X_test, Y_test, epochs=100, optimizer = optimizer)
y_predicted = network.predict(X_test)
network.save_model(model_file)
network.evaluate_prediction(Y_test, y_predicted)
for key, val in network.metrics.items():
    print(f"The {key} is : {val}")
print("Model saved")