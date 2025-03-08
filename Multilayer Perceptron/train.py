import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from optimizer import Optimizer
from network import Network
import sys
from load_data import load_data
import matplotlib.pyplot as plt

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
        optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
    case "momentum":
        optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.01, momentum = 0.9)
    case "adagrad":
        optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
    case "nesterov":
        optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.4)
    case "rmsprop":
        optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.001, beta = 0.9, epsilon = 1e-8)
    case "adam":
        optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8)
    case _:
        optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.01)
network = None
network = Network()
network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
network.add_layer(nodes = 20)
network.add_layer(nodes = 10)
network.add_layer(nodes = 5)
network.add_layer(nodes = 2, activation = "softmax")
arr_looses, arr_accuracies =network.train(X_train, Y_train, X_test, Y_test, epochs=1000,  optimizer = optimizer, verbose = True)
network.save_model(model_file)
#curr_accuracy =network[i].train(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=128, optimizer = optimizer)

print ("Metrics for the lastiteration:")
for key, value in network.metrics.items():
    print(f"{key}: {value:.4f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(arr_looses, label='Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(arr_accuracies, label='Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()