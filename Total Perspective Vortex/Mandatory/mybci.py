import sys
import os
from pipeline.pipeline import pipeline
from utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from pipeline.classifiers import Classifier
from utils.utils import msg_error, get_classifiers_list

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
list_files = create_list_files(subjects=subjects, runs=runs, root=root)

if list_files is None:
    print("No files to be downloaded")
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
my_classifier = Classifier()
my_classifier.set_classifier(classifier)
my_pippeline.make_pipeline(my_classifier.get_classifier())
train_results = my_pippeline.train_model(X_train, y_train)
test_results = my_pippeline.test_model(X_test, y_test)

print(f"Results are for train: {train_results}")
print(f"Results are for test: {test_results}")
