import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from optimizer import Optimizer
from network import Network
import sys
import copy
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

if len(sys.argv) == 4:
    model_optimizer = sys.argv[3]
else:
    model_optimizer = "sgd"


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
match model_optimizer:
    case "sgd":
        optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.01)
    case "momentum":
        optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.01, momentum = 0.9)
    case "adagrad":
        optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
    case "adagrad":
        optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.9)
    case "rmsprop":
        optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.001, beta = 0.9, epsilon = 1e-8)
    case "adam":
        optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-8)
    case _:
        optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.01)
best_network ={}
for i in range (0,10):
    curr_best_network = None
    last_accuracy = 0
    for k in range(5 , 10):
        for j in range(10,20):
            for l in range(20,25):
                print (f"Training layer 2: {l} nodes, layer 3: {j} nodes and layer 4 {k} nodes")
                network = None
                network = Network()
                network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
                network.add_layer(nodes = l)
                network.add_layer(nodes = j)
                network.add_layer(nodes = k)
                #networ.add_layer(nodes = 1)
                network.add_layer(nodes = 2, activation = "softmax")
                curr_accuracy =network.train(X_train, Y_train, X_test, Y_test, epochs=100,  optimizer = optimizer)
                #curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=128, optimizer = optimizer)
                if curr_accuracy > last_accuracy:
                    print (f"Accuracy improved from {last_accuracy} to {curr_accuracy} with layer 2: {l} nodes, layer 3: {j} nodes and layer 4 {k} nodes")
                    last_accuracy = curr_accuracy
                    curr_best_network = [l, j, k, curr_accuracy]
    best_network[i] = curr_best_network

print (f"Best network in each round were:")
for _, values in best_network.items():
    print (values)
    #print(f"Accuracy {values[3]} with layer 2: {values[0]} nodes, layer 3: {values[1]} nodes and layer 4 {values[2]} nodes")
last_network = None
last_accuracy = 0
for index, values in best_network.items():
    print(values)
    network = None
    network = Network()
    network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
    network.add_layer(nodes = values[0])
    network.add_layer(nodes = values[1])
    network.add_layer(nodes = values[2])
    network.add_layer(nodes = 2, activation = "softmax")
    network.train(X_train, Y_train, X_test, Y_test, epochs=100, optimizer = optimizer)
    y_predicted = network.predict(X_test)
    if curr_accuracy > last_accuracy:
        print (f"Accuracy improved from {last_accuracy} to {curr_accuracy} with layer 2: { values[0]} nodes, layer 3: { values[1]} nodes and layer 4 { values[2]} nodes")
        last_accuracy = curr_accuracy
        last_network = [values[0], values[1], values[2], curr_accuracy]

network = None
network = Network()
network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
network.add_layer(nodes = last_network[0])
network.add_layer(nodes = last_network[1])
network.add_layer(nodes = last_network[2])
network.add_layer(nodes = 2, activation = "softmax")
network.train(X_train, Y_train, X_test, Y_test, epochs=100, optimizer = optimizer)
network.save_model(model_file)
network.evaluate_prediction(Y_test, y_predicted)
for key, val in network.metrics.items():
    print(f"The {key} is : {val}")
print("Model saved")