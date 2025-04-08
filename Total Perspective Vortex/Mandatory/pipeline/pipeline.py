import numpy as np
import pandas as pd
import os
import random
from mne import  channels, io, events_from_annotations, Epochs, time_frequency
from .event_type import Event_Type
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from mne import io, Epochs, events_from_annotations
from mne.datasets import sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from reshape_transformer import ReshapeTransformer
from Debugger import DebugTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
class pipeline():
    def __init__(self, folder = None, filename=None):
        self.weights = None
        self.files = []
        self.ch_names = None
        self.montage = None
        self.lfreq = 0.1
        self.hfreq = 40
        self.n_components = None
        self.pipeline = None
        self.csp = None
        self.learning = None

        if folder is not None:
            self.add_files_from_folder(folder)
        if filename is None:
            self.add_file(filename)

    def add_files_from_folder(self, folder = None):
        """
        Add all files from a folder to the pipeline for processing.
        Parameters:
        folder (str): Path to the folder containing files to be added.
        """
        for ruta, carpeta, archivos in os.walk(folder):
            for archivo in archivos:
                self.add_file(os.path.join(ruta, archivo))
    
    def add_file(self, filename = None):
        """
        Add file to the pipeline for processing.
        If the file is a valid EDF and contains events, it will be added to the list of files.
        If the file is not valid or does not contain events, it will be ignored.
        Parameters:
        filename (str): Path to the file to be added.
        """
        if filename is None:
            return
        if os.path.exists(filename) and filename.endswith(".edf"):
#            self.open_raw(filename)
#            events, event_id = events_from_annotations(self.raw)
#            if events is None or len(events) == 0:
#                return
            self.files.append(filename)
        return
    
    def config_montage(self, excluded_channels = None, n_components = None):
        # Step 1: Define the standard 10-10 montage
        self.montage = channels.make_standard_montage("standard_1020")
        # Step 2: Remove excluded channels
        self.ch_names = [ch for ch in self.montage.ch_names if ch not in excluded_channels]
        if n_components is None:
            print("WARNING: Number of components not defined.Fit function may fail.")
        else:
            self.n_components = n_components
        return 
    
    def set_filter_freq (self, lfreq = 0.1, hfreq = 40):
        self.lfreq = lfreq
        self.hfreq = hfreq
        return

    def open_raw(self, file):
        """
        Open a raw file and apply the montage and filter settings.
        Parameters:
        file (str): Path to the raw file to be opened.
        """
        # Open the raw file using MNE-Python
        self.raw = io.read_raw_edf(file,preload=True)
        self.raw.rename_channels({old: new for old, new in zip(self.raw.ch_names, self.ch_names)})
        self.raw.set_montage(self.montage)
        self.raw = self.raw.filter(l_freq = self.lfreq, h_freq = self.hfreq , fir_design='firwin')
        # Seleccionar solo los canales de EEG
        self.raw.pick_types(eeg=True)

    def define_test_train(self, percentage = None, mask = None):
        """
        Define the test and train sets based on the provided percentage or mask.
        Parameters:
        percentage (float): Percentage of data to be used for training.
        mask (array-like): Boolean mask indicating which files to use for training. 
        """
        train_model = []
        test_model = []
        if mask is None and percentage is None:
            percentage = 0.8
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.array(np.random.rand(len(self.files)) < percentage)
            
        for i in range(len(self.files)):
            if self.mask[i]:
                train_model.append(self.files[i])
            else:
                test_model.append(self.files[i])
        return train_model, test_model

    """
            tmin = -0.2  # 200 ms before the event
            tmax = 4   # 500 ms after the event
            #event_types = Event_Type(file)
            self.epochs = Epochs(self.raw, self.events, self.event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
            # Example: Compute the power spectral density (PSD) for each epoch
            #spectrum= self.raw.compute_psd(method="welch", fmin=4, fmax=40)
            #psds, freqs = spectrum.get_data(return_freqs=True)
            # You can select specific frequency bands for further analysis
            # Example: Extract alpha and beta band power
            #alpha_band = (8, 12)  # Alpha band (8-12 Hz)
            #beta_band = (13, 30)  # Beta band (13-30 Hz)

            #alpha_power = psds[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])]
            #beta_power = psds[:, (freqs >= beta_band[0]) & (freqs <= beta_band[1])]
            # Convertir a numpy para CSP
            X_curr = self.epochs.get_data()  # (n_trials, n_channels, n_samples)
            #self.X  = np.hstack([alpha_power, beta_power])

            Y_curr = self.epochs.events[:, -1]  # Etiquetas
            self.X, self.Y = self.csp_model.fill(X_curr, Y_curr)
        return self.csp_model
    """

    def generate_data(self, files):
        """
        Get the training data from the raw file.
        Returns:
        tuple: Tuple containing the training data (X) and labels (Y).
        """
        labels = []
        data = []
        for file in files:
            self.open_raw(file)
            events, event_id = events_from_annotations(self.raw)
            if events is None or len(events) == 0:
                continue
            tmin = -0.2  # 200 ms before the event
            tmax = 4
            epochs = Epochs(self.raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
            curr_label = epochs.events[:, -1]
            if len(curr_label) == 0:
                continue
            data.append(epochs.get_data())
            event_type = Event_Type(filename = file)
            labels.append(event_type.convert_event_labels(curr_label))
        X = np.vstack(data)
        Y = np.stack(labels).flatten()
        mask = np.array(np.random.rand(len(Y)) < 0.5)
        mask1 = (Y==0)
        mask = mask & mask1
        X = X[~mask]
        Y = Y[~mask]
        return X, Y

    def train_model(self,X,y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        self.test_model(X_val,y_val)

    def test_model(self, X, y):
        y_pred = self.pipeline.predict(X)
        self.evaluate_prediction(y, y_pred)
        return 
    
    def evaluate_prediction(self, Y, y_pred):
        print("Classification report:")
        print(classification_report(Y, y_pred)) 
        print("Accuracy score:")
        print(accuracy_score(Y, y_pred))
        print("Precision, recall, f1-score:")
        precision, recall, f1_score, _ = precision_recall_fscore_support(Y, y_pred, average='weighted')
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return
    """("estimator", GeneralizingEstimator(
                LinearModel(SVC(kernel='poly', C=1, gamma='scale', probability=True)),
                scoring="accuracy",
                n_jobs=1,
                verbose=True,
            ))
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
            """
    def make_pipeline(self):
        params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        
        #("csp",CSPModel (n_components = self.n_components)),
        self.csp = CSP (n_components = 5, reg = None, log = None, transform_into = "average_power", rank = {'eeg':64}, norm_trace = False)
        #self.learning = GridSearchCV(SVC(kernel='rbf', gamma=0.5, C=0.1), params_grid, cv=9)
        self.learning = SVC(kernel='rbf', C=0.1, gamma=0.5, probability=True)
        #self.learning = RandomForestClassifier(n_estimators=100, random_state=42)
        #self.learning = KernelPCA(n_components=5, kernel='rbf', gamma=0.5)
        #self.learning = LinearDiscriminantAnalysis()
        #self.learning = LinearModel(SVC(kernel='rbf', C=1, gamma='scale', probability=True))
        #self.learning = GeneralizingEstimator(SVC(kernel='rbf', C=1, gamma='scale', probability=True), scoring="accuracy", n_jobs=1, verbose=True)

        self.pipeline = Pipeline([
            ("csp",self.csp),
            #('reshape',ReshapeTransformer()),
            ("scaler", StandardScaler()),
            #('classifier', LinearDiscriminantAnalysis())
            ('classifier',self.learning)
            ])
        return pipeline
    
    def save_weights(self, filename):
        self.csp_model.save_model(filename)
    
    def get_dataset_train(self):
        return self.csp_model_train.get_dataset()

    def save_dataset_train(self, filename):
        self.csp_model_train.save_dataset(filename)
