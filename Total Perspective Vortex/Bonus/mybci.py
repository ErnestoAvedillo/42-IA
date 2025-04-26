import sys
import os
import numpy as np
from ..pipeline.pipeline import pipeline
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from ..utils.neural_network_class.network import Network
from ..utils.neural_network_class.layer import Layer
from ..utils.neural_network_class.optimizer import Optimizer
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

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

#root = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/"
root = "/home/eavedill/sgoinfre/mne_data/files/"
list_files = create_list_files(subjects=subjects, runs=runs, root=root)

if list_files is None or len(list_files) == 0:
    print("No files opened")
    sys.exit(1)
my_pippeline = pipeline()
my_pippeline.config_montage(n_components = 5)

for item in list_files:
    if os.path.isdir(item):
        my_pippeline.add_files_from_folder(folder = item)
    else:
        my_pippeline.add_file(filename = item)

train_model, test_model  = my_pippeline.define_test_train(percentage=0.8)
X_train, y_train = my_pippeline.generate_data(train_model)
X_test, y_test = my_pippeline.generate_data(test_model)
csp = CSP(n_components=63, reg=None, log=True, norm_trace=False)
csp.fit(X_train, y_train)
X_train = csp.transform(X_train)
X_test = csp.transform(X_test)
#X_train = X_train.reshape(-1, 1, 7, 7)
#X_test = X_test.reshape(-1, 1, 7, 7)
outputs = np.unique(y_train)
output_len = len(np.unique(y_train))
y_train_NN = np.zeros((y_train.shape[0], output_len))
y_test_NN = np.zeros((y_test.shape[0], output_len))
for i in range(output_len):
    y_train_NN[y_train == outputs[i], i] = 1
    y_test_NN[y_test == outputs[i], i] = 1
X_train, X_val, y_train_NN, y_val_NN = train_test_split(X_train, y_train_NN, test_size=0.2, random_state=42)
network = Network()
network.add_layer(layer = Layer(layer_type = "dense", input_shape = 128, data_shape=X_train.shape[1]))
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
network.train(X_train, y_train_NN, X_val, y_val_NN, epochs=200,  optimizer = optimizer, verbose = True)

network.plot_accuracies()
network.plot_losses()
y_pred = network.predict(X_train, y_train_NN)
print(classification_report(y_train_NN, y_pred))
print(f"Accuracy train : {accuracy_score(y_train_NN, y_pred)}")
y_pred = network.predict(X_val, y_val_NN)
print(classification_report(y_val_NN, y_pred))
print(f"Accuracy val: {accuracy_score(y_val_NN, y_pred)}")
y_pred = network.predict(X_test, y_test_NN)
print(classification_report(y_test_NN, y_pred))
print(f"Accuracy test: {accuracy_score(y_test_NN, y_pred)}")