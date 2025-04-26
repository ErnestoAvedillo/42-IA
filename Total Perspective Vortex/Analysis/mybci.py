import sys
import os
import ast  # Abstract Syntax Trees
from ..pipeline.process_data import ProcessData
from ..pipeline.pipeline import My_Pipeline
from ..utils.create_list_files import create_list_files
from ..pipeline.classifiers import Classifier

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

if list_files is None:
    print("No files to be downloaded")
    sys.exit(1)
my_process_data = ProcessData()
my_process_data.config_montage(n_components = 5)

for item in list_files:
    if os.path.isdir(item):
        my_process_data.add_files_from_folder(folder = item)
    else:
        my_process_data.add_file(filename = item)

train_model, test_model  = my_process_data.define_test_train(percentage=0.8)
X_train, y_train = my_process_data.generate_data(train_model)
X_test, y_test = my_process_data.generate_data(test_model)
my_classifier = Classifier()
results = {}
for classifier in my_classifier.get_dict_keys():
    my_classifier.set_classifier(classifier)
    print(f"Classifier: {classifier} --> Classifier type: {my_classifier.get_classifier_type(classifier)}")
    my_pipeline = My_Pipeline()
    my_pipeline.make_pipeline(my_classifier.get_classifier())
    train_results = my_pipeline.train_model(X_train, y_train)
    test_results = my_pipeline.test_model(X_test, y_test)
    results [classifier] = [train_results, test_results]

print(f"|Clasifier | Validation precision | Test Precision|")
print(f"|----------|----------------------|---------------|")
for key , result in results.items():
    print(f"| {key} | {result[0][0]} | {result[1][0]} |")
