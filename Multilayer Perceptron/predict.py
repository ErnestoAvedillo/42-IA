import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
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
    sys.exit(1)

print (data.keys())
#input("Press enter..")

#Y = data["1_B"].to_numpy().astype(int)
Y = data[["1_B","1_M"]].to_numpy().astype(int)
#X = data.drop(columns=['2', '3', '4', '5','15','22', '23','24', '25',"1_B"]).to_numpy().astype(float)
#X = data.drop(columns=['4', '5','25',"1_B","1_M"]).to_numpy().astype(float)
X = data.drop(columns=["1_B","1_M"]).to_numpy().astype(float)
#X = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
for i in range(X_train.shape[1]):
    print("Column", i)
    print(f"Mean    train: {X_train[:,i].mean()} Test: {X_test[:,i].mean()}")
    print(f"Std_dev train: {X_train[:,i].std()} Test: {X_test[:,i].std()}")
input("Press enter...")

network = {}
i = 0
last_accuracy = 0
for k in range(5 , 10):
    for j in range(10,20):
        print (f"Training with {j} and {k} nodes in layers 2 and 3")
        network[i] = Network()
        network[i].add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
        network[i].add_layer(nodes = j)
        network[i].add_layer(nodes = k)
        #network[i].add_layer(nodes = 1)
        network[i].add_layer(nodes = 2, activation = "softmax")
        curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100)
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
network.add_layer(nodes = 1)
network.train(X_train, Y_train, X_test, Y_test, epochs=100)
network.save_model("model.json")
print("Model saved")