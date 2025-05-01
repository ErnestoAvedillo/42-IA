import sys
import os
from ..pipeline.pipeline import My_Pipeline
from ..pipeline.process_data import ProcessData
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from ..pipeline.classifiers import Classifier
from ..utils.utils import msg_error, get_classifiers_list

if len(sys.argv) < 3:
    msg_error()
    sys.exit(1)

# Get the argument (excluding the script name itself)
arg = sys.argv[1]
# Convert string to list
subjects = ast.literal_eval(arg)  # Safer than eval()
# Get the argument (excluding the script name itself)
arg = sys.argv[2]
# Convert string to list
runs = ast.literal_eval(arg)  # Safer than eval()

classifier = sys.argv[3]
classifiers = get_classifiers_list().keys()
if classifier not in classifiers:
    print(f"The classifier {classifier} is not implemented. See below the accepted list.")
    for item in classifiers:
        print(item)
root = "/home/ernesto/mne_data/physionet/files/eegmmidb/1.0.0/"
#root = "/home/eavedill/sgoinfre/mne_data/files/"
list_files = create_list_files(subjects=subjects, runs=runs, root=root)

if list_files is None or len(list_files) == 0:
    print("No files opened")
    sys.exit(1)
my_process_data = ProcessData()
excluded_channels = ['AF9', 'AF10','AF5', 'AF1','AF2', 'AF6','F9', 'F10','FT9', 'FT10','A1', 'A2','M1', 'M2','TP9', 'TP10','P9', 'P10','PO5', 'PO1','PO2', 'PO6','PO9', 'PO10','O9', 'O10']
my_process_data.config_montage(excluded_channels=excluded_channels)

for item in list_files:
    if os.path.isdir(item):
        my_process_data.add_files_from_folder(folder = item)
    else:
        my_process_data.add_file(filename = item)

train_model, test_model  = my_process_data.define_test_train(percentage=0.8)
X_train, y_train = my_process_data.generate_data(train_model)
X_test, y_test = my_process_data.generate_data(test_model)
my_classifier = Classifier()
my_classifier.set_classifier(classifier)
my_pipeline = My_Pipeline()
my_pipeline.make_pipeline(my_classifier.get_classifier())
train_results = my_pipeline.train_model(X_train, y_train)
test_results = my_pipeline.test_model(X_test, y_test)

print(f"Results are for train: {train_results}")
print(f"Results are for test: {test_results}")
