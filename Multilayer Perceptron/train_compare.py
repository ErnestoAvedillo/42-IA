import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from layer import Layer
from optimizer import Optimizer
from network import Network
import sys
from load_data import load_data
import matplotlib.pyplot as plt
def get_optimizer(optimizer = "sgd"):
    match model_optimizer:
        case "sgd":
            optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
        case "momentum":
            optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.001, momentum = 0.9)
        case "adagrad":
            optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
        case "nesterov":
            optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.4)
        case "rmsprop":
            optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.01, beta = 0.4, epsilon = 1e-8)
        case "adam":
            optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8)
        case _:
            optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
    return optimizer

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

plt.figure(figsize=(10, 5))
ax1 = plt.subplot(2, 2, 1)
ax1.set_title('Loss test over epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

ax2 = plt.subplot(2, 2, 2)
ax2.set_title('Accuracy test over epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

ax3 = plt.subplot(2, 2, 3)
ax3.set_title('Loss validation  over epochs')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy')

ax4 = plt.subplot(2, 2, 4)
ax4.set_title('Accuracy validation over epochs')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Accuracy')


optimizers = ["sgd", "momentum", "adagrad", "nesterov", "rmsprop", "adam"]
for optimizer_type in optimizers:
    optimizer = get_optimizer(optimizer_type)
    network = None
    network = Network()
    network.add_layer(layer = Layer(nodes = 30, input_dim=X_train.shape[1]))
    network.add_layer(nodes = 20)
    network.add_layer(nodes = 10)
    network.add_layer(nodes = 5)
    network.add_layer(nodes = 2, activation = "softmax")
    arr_looses, arr_accuracies, arr_looses_val, arr_accuracies_val =network.train(X_train, Y_train, X_test, Y_test, epochs=1000,  optimizer = optimizer, verbose = True)
    network.save_model(model_file)
    ax1.plot(arr_looses, label='Loss ' + optimizer_type)
    ax2.plot(arr_accuracies, label='Accuracy ' + optimizer_type)
    ax3.plot(arr_looses_val, label='Loss validation ' + optimizer_type)
    ax4.plot(arr_accuracies_val, label='Accuracy validation ' + optimizer_type)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.tight_layout()
plt.show()