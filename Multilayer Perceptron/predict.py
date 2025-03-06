import pandas as pd
import numpy as np
from network import Network
import sys
from load_data import load_data

if len(sys.argv) < 3: 
    print("Usage: python predict.py dataset.csv model.json")
    sys.exit(1)
try:
    data = load_data(sys.argv[1],header = 0)
except FileNotFoundError:
    print(f"File {sys.argv[1]} not found")
    sys.exit(0)

mynetwork = Network()
try:
    mynetwork.open_model(sys.argv[2])
except:
    print(f"File {sys.argv[2]} does not exist.")
    sys.exit(0)
    
Y = data[["1_B","1_M"]].to_numpy().astype(int)
X = data.drop(columns=["1_B","1_M"]).to_numpy().astype(float)
y_predicted = mynetwork.predict(X)
mynetwork.evaluate_prediction(Y, y_predicted)
for key, val in mynetwork.metrics.items():
    print(f"The {key} is : {val}")