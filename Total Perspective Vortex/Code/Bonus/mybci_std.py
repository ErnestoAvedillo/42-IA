import sys
import os
import numpy as np
from ..pipeline.process_data import ProcessData
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from sklearn.model_selection import train_test_split
from mne.decoding import CSP, cross_val_multiscore
from sklearn.metrics import classification_report, accuracy_score
from .cov import CalculateCovariance as cov, Normalize as norm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from .NN import SmallCSPCNN
from ..pipeline.reshape_transformer import ReshapeTransformer
import joblib

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
from sklearn.base import TransformerMixin, BaseEstimator

class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Reshape X of shape (n_samples, *dims) → (n_samples, prod(dims))."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        n = X.shape[0]
        return X.reshape(n, -1)
train_model, test_model  = my_process_data.define_test_train(percentage=0.80)
X_train, y_train = my_process_data.generate_data(train_model)
X_test, y_test = my_process_data.generate_data(test_model)
#csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp = CSP(n_components=8, reg="ledoit_wolf", log=None, rank="full", transform_into="csp_space")

csp.fit(X_train, y_train)
X_train = csp.transform(X_train)
X_test = csp.transform(X_test)  

#X_train, X_val, y_train_NN, y_val_NN = train_test_split(X_train, y_train_NN, test_size=0.5, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f"Data fitted and dvided in: Train {X_train.shape} and test {X_test}")

clf = make_pipeline(
    csp,                         # yields (n_trials, n_components) features
    ReshapeTransformer(),
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(50, ), 
                  activation='relu',
                  alpha=1e-3,      # L2 regularization
                  max_iter=200)
)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
scores = cross_val_multiscore(clf, X_train, y_train, cv=cv, n_jobs=None)
scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=None)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Accuracy score:")
print(accuracy_score(y_test, y_pred))
print("Precision, recall, f1-score:")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
results = [precision, recall, f1_score]
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")


joblib.dump(clf, 'bci_Bonus.pkl')