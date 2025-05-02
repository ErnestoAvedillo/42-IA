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


root = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/"
#root = "/home/eavedill/sgoinfre/mne_data/files/"
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

test_model, _  = my_process_data.define_test_train(percentage=1)
X_test, y_test = my_process_data.generate_data(test_model)
with open("bci_Bonus.json",'r') as archivo:
    model = json.load(archivo)
try:
    type = model.get("type")
    if type== "csp":
        csp_model = model.get("csp",None)
        csp_patterns = model.get("csp_patterns",None)
        csp_filters = model.get("csp_filters",None)
    output_len = model.get("output_len")
    outputs = model.get("outputs")
    network_model = model.get("network")
except:
    raise ValueError("Invalid model file.")

if type == "cov":
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
elif type == "norm":
    X_test, _ = norm(X_test[:,:,:160], y_test)
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
elif type == "csp":
    csp = CSP(n_components=63, reg=None, log=True, norm_trace=False)
    csp.set_params(**csp_model)
    csp.filters_ = np.array(csp_filters)
    csp.patterns_ = np.array(csp_patterns)
    X_test = csp.transform(X_test)  
else:
    raise ValueError("Invalid type. Choose 'cov', 'norm', or 'csp'.")
y_test_NN = np.zeros((y_test.shape[0], output_len))
for i in range(output_len):
    y_test_NN[y_test == outputs[i], i] = 1

network = Network()
network.set_model(network_model)
y_pred, metrics = network.predict(X_test, y_test_NN)
print(classification_report(y_test_NN, y_pred))
print(f"Accuracy test: {accuracy_score(y_test_NN, y_pred)}")
