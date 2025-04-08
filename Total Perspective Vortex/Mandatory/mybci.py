import sys
import os
from pipeline.pipeline import pipeline

if len(sys.argv) < 2:
    sample_data = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/S081/"
    #print("plese enter file to be analised.")
    #sys.exit(1)

my_pippeline = pipeline()
excluded_channels = ["AF1","AF2", "AF5", "AF6", "AF9", "AF10", "F9", "F10", "FT9", "FT10", "A1", "A2", "M1", "M2", "TP9", "TP10", "P9", "P10", "PO1", "PO2", "PO5", "PO6", "PO9", "PO10", "O9", "O10"]
my_pippeline.config_montage(excluded_channels=excluded_channels, n_components = 5)

for i in range(1,len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        my_pippeline.add_files_from_folder(folder = sys.argv[i])
    else:
        my_pippeline.add_file(folder = sys.argv[i])

train_model, test_model  = my_pippeline.define_test_train(percentage=0.8)
X_train, y_train = my_pippeline.generate_data(train_model)
X_test, y_test = my_pippeline.generate_data(test_model)
my_pippeline.make_pipeline()
my_pippeline.train_model(X_train, y_train)
my_pippeline.test_model(X_test, y_test)

