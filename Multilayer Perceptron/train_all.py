import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from optimizer import Optimizer
from network import Network
import sys
import copy
from load_data import load_data
import os

if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1],header = 0)
except FileNotFoundError:
    print("File not found")
    sys.exit(0)

if len(sys.argv) == 3:
    folder = sys.argv[2]
else:
    folder = "./models"

if not os.path.exists(folder):
    os.makedirs(folder)

#get data and split it
Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=["1_B","1_M"]).to_numpy().astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

network = {}
i = 0
last_accuracy = 0
models_optimizer ={
     "sgd": Optimizer(optimizer = "sgd", learning_rate = 0.01),
     "momentum": Optimizer(optimizer = "momentum", learning_rate = 0.01, momentum = 0.9),
     "adagrad": Optimizer(optimizer = "adagrad", learning_rate = 0.01),
     "nesterov": Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.9),
     "rmsprop": Optimizer(optimizer = "rmsprop", learning_rate = 0.001, beta = 0.9, epsilon = 1e-8),
     "adam": Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-8)
    }
summary_best = {}
for keys, optimizer in models_optimizer.items():
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
                    _, curr_accuracy, loss_val, curr_accuracy_val =network.train(X_train, Y_train, X_test, Y_test, epochs=100,  optimizer = optimizer)
                    #curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=128, optimizer = optimizer)
                    if curr_accuracy[-1] > last_accuracy:
                        print (f"Accuracy improved from {last_accuracy} to {curr_accuracy} with layer 2: {l} nodes, layer 3: {j} nodes and layer 4 {k} nodes")
                        last_accuracy = curr_accuracy[-1]
                        curr_best_network = [l, j, k, curr_accuracy ,"name", copy.deepcopy(network)]
        curr_best_network[4] = folder + f"/model_Nr_{keys}_{i}_{curr_best_network[0]}_{curr_best_network[1]}_{curr_best_network[2]}.json"
        curr_best_network[5].save_model(curr_best_network[4])
        best_network[i] = curr_best_network[:5]
    summary_best[keys] = best_network

print (f"Best network in each round were:")

columns = ["optimizer", "iteration", "layer_2", "layer_3", "layer_4", "accuracy", "file_name"]
summary_df = pd.DataFrame(columns=columns)
for optimizer, best_network in summary_best.items():
    for iteration, values in best_network.items():
        new_raw = pd.DataFrame([{
            "optimizer": optimizer,
            "iteration": iteration,
            "layer_2": values[0],
            "layer_3": values[1],
            "layer_4": values[2],
            "accuracy": values[3],
            "file_name": values[4]
        }])
        summary_df = pd.concat([summary_df, new_raw], ignore_index=True)  
summary_df.to_csv(folder + "/summary.csv")
print(summary_df)
