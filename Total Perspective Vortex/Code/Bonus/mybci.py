import sys
import os
import numpy as np
from ..pipeline.process_data import ProcessData
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from ..utils.neural_network_class.network import Network
from ..utils.neural_network_class.layer import Layer
from ..utils.neural_network_class.optimizer import Optimizer
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from .cov import CalculateCovariance as cov, Normalize as norm
import json

if len(sys.argv) < 2:
    print("plese enter list to be analysed.")
    sys.exit(1)

# Get the argument (excluding the script name itself)
arg = sys.argv[1]
# Convert string to list
subjects = ast.literal_eval(arg)  # Safer than eval()
# Get the argument (excluding the script name itself)
arg = sys.argv[2]
# Convert string to list
runs = ast.literal_eval(arg)  # Safer than eval()

# Get the argument (excluding the script name itself)
type = sys.argv[3]

root = os.getenv('MNE_DATA')
print (f"searching data in folder {root}")

list_files = create_list_files(subjects=subjects, runs=runs, root=root)

if list_files is None or len(list_files) == 0:
    print("No files opened")
    sys.exit(1)
my_process_data = ProcessData()
excluded_channels = ['AF9', 'AF10','AF5', 'AF1','AF2', 'AF6','F9', 'F10','FT9', 'FT10','A1', 'A2','M1', 'M2','TP9', 'TP10','P9', 'P10','PO5', 'PO1','PO2', 'PO6','PO9', 'PO10','O9', 'O10']
my_process_data.config_montage(n_components = 5, excluded_channels = excluded_channels)

for item in list_files:
    if os.path.isdir(item):
        my_process_data.add_files_from_folder(folder = item)
    else:
        my_process_data.add_file(filename = item)

train_model, test_model  = my_process_data.define_test_train(percentage=0.80)
X_train, y_train = my_process_data.generate_data(train_model)
X_test, y_test = my_process_data.generate_data(test_model)
if type == "cov":
    X_train= cov(X_train[:,:,:160], y_train, ddof=0)
    X_test= cov(X_test[:,:,:160], y_test, ddof=0)
    X_train = X_train.reshape(-1, 1, X_train.shape[1], X_test.shape[2])
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
elif type == "norm":
    X_train, _ = norm(X_train[:,:,:160], y_train)
    X_test, _ = norm(X_test[:,:,:160], y_test)
    X_train = X_train.reshape(-1, 1, X_train.shape[1], X_test.shape[2])
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
elif type == "csp":
    #csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    input("Press Enter to continue0...")
    csp = CSP(n_components=8, reg=None, log=None, rank="full", transform_into="csp_space")
    input("Press Enter to continue1...")

    csp.fit(X_train, y_train)
    input("Press Enter to continue2...")
    X_train = csp.transform(X_train)
    print(f"Fitting CSP with {X_train.shape[1]} channels and {X_train.shape[2]} samples")
    input("Press Enter to continue3...")
    X_test = csp.transform(X_test)  
else:
    raise ValueError("Invalid type. Choose 'cov', 'norm', or 'csp'.")
input("Press Enter to continue3...")
outputs = np.unique(y_train)
output_len = len(np.unique(y_train))
y_train_NN = np.zeros((y_train.shape[0], output_len))
y_test_NN = np.zeros((y_test.shape[0], output_len))
for i in range(output_len):
    y_train_NN[y_train == outputs[i], i] = 1
    y_test_NN[y_test == outputs[i], i] = 1
#X_train, X_val, y_train_NN, y_val_NN = train_test_split(X_train, y_train_NN, test_size=0.5, random_state=42)
X_test, X_val, y_test_NN, y_val_NN = train_test_split(X_test, y_test_NN, test_size=0.5, random_state=42)

network = Network()
if type == "csp":
    network.add_layer(layer_type = "input", data_shape = X_train.shape[1:])
    network.add_layer(layer_type = "dense", input_shape = 160)
else:
    network.add_layer(layer_type = "input", filters=1, data_shape = X_train.shape[1:])
    network.add_layer(layer_type='conv', kernel_size=4, activation='relu')
    network.add_layer(layer_type='max_pool', kernel_size=4, activation='relu')
    network.add_layer(layer_type='flattend')
network.add_layer(layer_type = "dense", input_shape = 64)
network.add_layer(layer_type = "dense", input_shape = 32)
network.add_layer(layer_type = "dense", input_shape = 16)
network.add_layer(layer_type = "dense", input_shape = output_len, activation = "softmax")
#optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
#optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.001, momentum = 0.9)
#optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
#optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.4)
#optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.01, beta = 0.4, epsilon = 1e-8)
optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8)
network.fit(X_train, y_train_NN, X_val, y_val_NN, epochs=500,  optimizer = optimizer, verbose = True)

network.plot_accuracies()
network.plot_losses()
y_pred, metrics  = network.predict(X_train, y_train_NN)
print(classification_report(y_train_NN, y_pred))
print(f"Accuracy train : {accuracy_score(y_train_NN, y_pred)}")
y_pred, metrics = network.predict(X_val, y_val_NN)
print(classification_report(y_val_NN, y_pred))
print(f"Accuracy val: {metrics['accuracy']}")
y_pred, metrics = network.predict(X_test, y_test_NN)
print(classification_report(y_test_NN, y_pred))
print(f"Accuracy test: {accuracy_score(y_test_NN, y_pred)}")

if type == "csp":
    bci_model = {"type": type,
                "csp": csp.get_params(),
                "csp_filters": csp.filters_.tolist(),
                "csp_patterns": csp.patterns_.tolist(),
                "csp_means": csp.mean_.tolist(),
                "csp_std": csp.std_.tolist(),
                "output_len": output_len,
                "outputs": outputs.tolist(),
                "network": network.get_model()}
else:
    bci_model = {"type": type,
                "output_len": output_len,
                "outputs": outputs.tolist(),
                "network": network.get_model()}
with open("bci_Bonus.json", "w") as archivo:
    json.dump(bci_model, archivo)
