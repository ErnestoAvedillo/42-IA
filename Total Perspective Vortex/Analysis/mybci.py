import sys
import os
from pipeline.pipeline import pipeline
from utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from pipeline.classifiers import Classifier

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
results = {}
for classifier in my_classifier.get_dict_keys():
    my_classifier.set_classifier(classifier)
    print(f"Classifier: {classifier} --> Classifier type: {my_classifier.get_classifier_type(classifier)}")
    my_pippeline.make_pipeline(my_classifier.get_classifier())
    train_results = my_pippeline.train_model(X_train, y_train)
    test_results = my_pippeline.test_model(X_test, y_test)
    results [classifier] = [train_results, test_results]

print(f"|Clasifier | Validation precision | Test Precision|")
print(f"|----------|----------------------|---------------|")
for key , result in results.items():
    print(f"| {key} | {result[0][0]} | {result[1][0]} |")
