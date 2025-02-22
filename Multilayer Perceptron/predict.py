import pandas as pd
import numpy as np
#from histogram import show_all_histograms
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
print(data.keys())
for key in data.keys():
    if data[key].value_counts().count() > 10:
        continue
    print(key, data[key].unique())
    print(key, data[key].value_counts())
#show_all_histograms(data)