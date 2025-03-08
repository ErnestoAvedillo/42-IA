import pandas as pd
import numpy as np
from network import Network
import sys
import os
from load_data import load_data

if len(sys.argv) < 3: 
    print("Usage: python predict.py dataset.csv ./models")
    sys.exit(1)
try:
    data = load_data(sys.argv[1],header = 0)
except FileNotFoundError:
    print(f"Folder {sys.argv[1]} not found")
    sys.exit(0)


Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=["1_B","1_M"]).to_numpy().astype(float)
for dir, _, files in os.walk(sys.argv[2]):
    for file in files:
        model_file = dir+'/'+file
        print(f"Testing model file: {model_file}")
        mynetwork = None
        mynetwork = Network()
        mynetwork.open_model(model_file)
        y_predicted = mynetwork.predict(X, Y)
        for key, val in mynetwork.metrics.items():
            print(f"The {key} is : {val:.4f}")
