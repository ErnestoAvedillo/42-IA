import sys
import os
import numpy as np
from pipeline.pipeline import pipeline
from neural_network_class.network import Network
from neural_network_class.layer import Layer
from neural_network_class.optimizer import Optimizer
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    sample_data = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/S081/"
    #print("plese enter file to be analised.")
    #sys.exit(1)

my_pippeline = pipeline()
for i in range(1,len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        my_pippeline.add_files_from_folder(folder = sys.argv[i])
    else:
        my_pippeline.add_file(folder = sys.argv[i])

excluded_channels = ["AF1","AF2", "AF5", "AF6", "AF9", "AF10", "F9", "F10", "FT9", "FT10", "A1", "A2", "M1", "M2", "TP9", "TP10", "P9", "P10", "PO1", "PO2", "PO5", "PO6", "PO9", "PO10", "O9", "O10"]
my_pippeline.config_montage(excluded_channels=excluded_channels, n_components = 5)
my_pippeline.define_test_train(percentage=0.8)
my_pippeline.train_model()

my_pippeline.save_dataset_train("train.csv")
my_pippeline.save_dataset_test("test.csv")

data = my_pippeline.get_dataset_train()
Y = np.array([data[:, -1] == option for option in range(int(max(data[:,-1])) + 1)]).T.astype(int)
X = data[:, :-1]
X = X.reshape(-1, 1, 64, 64)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
print (f"Imprimo el shape de X {X.shape}")
print (f"Imprimo el shape de Y {Y.shape}")
network = None
network = Network()

#network.add_layer(layer = Layer(layer_type = "dense", input_shape = 64, data_shape=X_train.shape[1]))
network.add_layer(layer_type = "max_pool", data_shape = (64,64), kernel_size = 4, activation = "relu")
network.add_layer(layer_type = "conv", kernel_size = 8 , filters = 1, activation = "relu")
network.add_layer(layer_type = "flattend")
network.add_layer(layer_type = "dense", input_shape = 32)
network.add_layer(layer_type = "dense", input_shape = 16)
network.add_layer(layer_type = "dense", input_shape = 5, activation = "softmax")
#optimizer = Optimizer(optimizer = "sgd", learning_rate = 0.001)
#optimizer = Optimizer(optimizer = "momentum", learning_rate = 0.001, momentum = 0.9)
#optimizer = Optimizer(optimizer = "adagrad", learning_rate = 0.01)
#optimizer = Optimizer(optimizer = "nesterov", learning_rate = 0.001, momentum = 0.4)
#optimizer = Optimizer(optimizer = "rmsprop", learning_rate = 0.01, beta = 0.4, epsilon = 1e-8)
optimizer = Optimizer(optimizer = "adam", learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8)
arr_looses, arr_accuracies, arr_looses_val, arr_accuracies_val =network.train(X_train, Y_train, X_test, Y_test, epochs=1000,  optimizer = optimizer, verbose = True)

data_test = my_pippeline.get_dataset_test()
Y_test = np.array([data_test[:, -1] == option for option in range(int(max(data_test[:,-1])) + 1)]).T.astype(int)
X_test = data_test[:, :-1]
X_test = X_test.reshape(-1, 1, 64, 64)
print (f"Imprimo el shape de X {X.shape}")
print (f"Imprimo el shape de Y {Y.shape}")
Y_pred = network.predict(X_test)
metrics = network.evaluate_prediction(Y_pred, Y_test)

for i in range(len(Y_pred)):
    print(f"Expected: {Y_test[i]} Predicted: {Y_pred[i]}")
for key, value in metrics.items():
    print(f"{key}: {value}")

my_pippeline.save_weights("pipeline_weights.json")

network.save_model("model_train.json")
