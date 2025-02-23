import pandas as pd
import numpy as np
from histogram import show_all_histograms
from sklearn.model_selection import train_test_split

from layer import Layer
from network import Network
import sys

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=None, header=None)
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
print("imprimo los datos")
print(data)
print("imprimo la descripcion de los datos")
print(data.describe())
print("imprimo las keys  de los datos")
keys = data.keys()
print (keys)
#show_all_histograms(data)
for key in keys:
    if data[key].value_counts().count() > 10:
        continue
    print(key, data[key].unique())
    print(key, data[key].value_counts())
Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=[0,"1_B","1_M"]).to_numpy().astype(float)
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
network = Network()
network.add_layer(layer = Layer(nodes = 30, input_dim=30))
network.add_layer(nodes = 20)
network.add_layer(nodes = 10)
network.add_layer(nodes = 2)
network.train(X_train, Y_train, X_test, Y_test, epochs=100)