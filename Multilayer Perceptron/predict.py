import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from network import Network
import sys

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=None, header=0)
    data = data.dropna()
    data = data.drop_duplicates()
    data = pd.get_dummies(data)
    return data


if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1])
except FileNotFoundError:
    print("File not found")
    sys.exit(1)

print (data.keys())
#input("Press enter..")

Y = data["1_B"].to_numpy().astype(int)
#Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=["1_B"]).to_numpy().astype(float)
#X = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
#print(X)
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

network = {}
accuracies = {}
i = 0
for j in range(5,21):
    network[i] = Network()
    network[i].add_layer(layer = Layer(nodes = 30, input_dim=30))
    network[i].add_layer(nodes = j)
    #network.add_layer(nodes = 2, activation="softmax")
    network[i].add_layer(nodes = 1)
    acc = network[i].train(X_train, Y_train, X_test, Y_test, epochs=100)
    accuracies[i] = acc
    i += 1
print(accuracies)