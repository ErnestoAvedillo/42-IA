import sys
import os
from ..pipeline.pipeline import My_Pipeline
from ..pipeline.process_data import ProcessData
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from ..pipeline.classifiers import Classifier
from ..utils.utils import msg_error, get_classifiers_list
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import joblib

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




root = os.getenv('MNE_DATA')
print (f"searching data in folder {root}")

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
test_model, _ = my_process_data.define_test_train(percentage=1)
X_test, y_test = my_process_data.generate_data(test_model)

my_pipeline = joblib.load('bci_Bonus_std.pkl')

y_pred = my_pipeline.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Accuracy score:")
print(accuracy_score(y_test, y_pred))
print("Precision, recall, f1-score:")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
results = [precision, recall, f1_score]
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")


