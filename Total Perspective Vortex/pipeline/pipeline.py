import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from mne import  channels, io, events_from_annotations, Epochs, time_frequency, Annotations, concatenate_raws
from mne.preprocessing import ICA, create_eog_epochs
from .event_type import Event_Type
from .classifiers import Classifier
from Mandatory.CSPModel import CSPModel
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    PSDEstimator
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from mne.datasets import sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from reshape_transformer import ReshapeTransformer
from Debugger import DebugTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
class pipeline():
    def __init__(self, folder = None, filename=None):
        self.weights = None
        self.files = []
        self.ch_names = None
        self.montage = None
        self.lfreq = 1
        self.hfreq = 30
        self.n_components = None
        self.pipeline = None
        self.csp = None
        self.mycsp = None
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
    
    def config_montage(self, standard = "standard_1005", excluded_channels = None, n_components = None):
        # Step 1: Define the standard 10-10 montage
        self.montage = channels.make_standard_montage(standard)
        # Step 2: Remove excluded channels
        if excluded_channels is not None:
            self.ch_names = [ch for ch in self.montage.ch_names if ch not in excluded_channels]
        if n_components is None:
            print("WARNING: Number of components asigned to default value.")
            n_components = 4
        self.n_components = n_components
        return 
    
    def set_filter_freq (self, lfreq = 1, hfreq = 30):
        self.lfreq = lfreq
        self.hfreq = hfreq
        return

    def open_raw(self, file):
        """
        Open a raw file, apply the montage, filter data and change the event names.
        Parameters:
            file (str): Path to the raw file to be opened.
        Returns:
            raw (mne.io.Raw): The raw data object after applying montage and filter.
        """
        # Open the raw file using MNE-Python
        raw = io.read_raw_edf(file,preload=True)
        # Check if the file is valid and contains events
        events, _ = events_from_annotations(raw)
        if events is None or len(events[:,2]) <= 1:
            return None
        # Apply the montage to the raw data
        raw.rename_channels({old: new for old, new in zip(raw.ch_names, self.montage.ch_names)})
        raw.set_montage(self.montage)
        # Filter the data
        raw = raw.filter(l_freq = self.lfreq, h_freq = self.hfreq , fir_design='firwin')
        # Seleccionar solo los canales de EEG
        raw.pick_types(eeg=True)
        # Change the event names to match the event type
        event_type = Event_Type(filename = file)
        keys = list(event_type.event_type.keys())
        values = list(event_type.event_type.values())
        events[events[:, 2] == 1, 2] = values[0]
        events[events[:, 2] == 2, 2] = values[1]
        events[events[:, 2] == 3, 2] = values[2]
        inverted_event_id = event_type.get_inverted_event_labels()
        onsets = events[:, 0] / raw.info['sfreq']
        durations = np.zeros(events.shape[0])
        durations[1:] = (events[1:,0] - events[:-1,0]) / raw.info['sfreq']
        # create descriptions
        descriptions = [inverted_event_id[int(eid)] for eid in events[:, 2]]
        # Create new annotations
        new_annotations = Annotations(onset=onsets, duration=durations, description=descriptions)
        # Set them to the raw object
        raw.set_annotations(new_annotations)
        return raw

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
            percentage = 0.7
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.array(np.random.rand(len(self.files)) < percentage)
            while self.mask.sum() == 0 or self.mask.sum() == len(self.files):
                self.mask = np.array(np.random.rand(len(self.files)) < percentage)
        for i in range(len(self.files)):
            if self.mask[i]:
                train_model.append(self.files[i])
            else:
                test_model.append(self.files[i])
        return train_model, test_model

    def generate_data(self, files):
        """
        Get the training data from the raw file.
        Returns:
        tuple: Tuple containing the training data (X) and labels (Y).
        """
        raws = []
        for file in files:
            raw = self.open_raw(file)
            if raw is None:
                continue
            raws.append(raw)
        self.raw = concatenate_raws(raws)
        self.raw.set_eeg_reference(projection=True)
        # remove unwanted artifacts
        ica = ICA(n_components=0.95, random_state=97, max_iter=800)
        ica.fit(self.raw, decim=3, reject_by_annotation=True)
        eog_inds, scores = ica.find_bads_eog(self.raw, ch_name=['Fp1', 'Fp2', 'AF3', 'AF4'], measure = 'correlation', threshold = 0.85)  # use your frontal channel
        print (f"ICA EOG indices are {eog_inds}")
        print (f"ICA EOG scores are {scores}")
        ica.exclude = eog_inds  # mark for exclusion
        #ica.plot_components(inst=self.raw, title="ICA Components")
        self.raw = ica.apply(self.raw)
        events, event_id = events_from_annotations(self.raw)
        print (f"Event id is {event_id}")
        tmin = -0.2  # 200 ms before the event
        tmax = 4
        epochs = Epochs(self.raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
        X = epochs.get_data()
        Y = epochs.events[:, 2]
        # Remove the T0 events from the data 
        #mask1 = np.array(np.random.rand(len(Y)) < 0.5)
        mask = (Y==1)
        #mask = mask & mask1
        X = X[~mask]
        Y = Y[~mask]
        return X, Y

    def train_model(self,X,y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_to_plot = self.csp.fit_transform(X_train, y_train)
        MY_X_to_plot = self.mycsp.fit_transform(X_train, y_train)

#        self.csp.plot_patterns(self.raw.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
#
#        labels = np.unique(y)
#        colors = np.random.rand(len(labels), 3)
#        for label in labels:
#            mask =  y_train == label
#            X_to_plot_ = X_to_plot[mask]
#            Y_to_plot_ = y_train[ mask]
#            plt.scatter(X_to_plot_[0], X_to_plot_[1], color = colors[label])
#        plt.show()
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        self.scores = cross_val_multiscore(self.pipeline, X_train, y_train, cv=cv, n_jobs=None)
        #self.scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, n_jobs=None)
        print(f"Scores are {self.scores}")
        self.pipeline.fit(X_train, y_train)
        return self.test_model(X_val,y_val)

    def test_model(self, X, y):
        y_pred = self.pipeline.predict(X)
        return self.evaluate_prediction(y, y_pred)
        
    
    def evaluate_prediction(self, Y, y_pred):
        print("Classification report:")
        print(classification_report(Y, y_pred)) 
        print("Accuracy score:")
        print(accuracy_score(Y, y_pred))
        print("Precision, recall, f1-score:")
        precision, recall, f1_score, _ = precision_recall_fscore_support(Y, y_pred, average='weighted')
        results = [precision, recall, f1_score]
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return results

    def make_pipeline(self, classifier = None):
        self.learning = classifier
        #("csp",CSPModel (n_components = self.n_components)),
        #self.csp = CSP (n_components = 4, reg = None, log = None, transform_into = "average_power", rank = {'eeg':64}, norm_trace = False)
        self.csp = CSP (n_components = 4, reg = None, log = True, norm_trace = False)
        self.mycsp = CSPModel (n_components = 4)
        self.pipeline = Pipeline([
            ("csp",self.csp),
            #('reshape',ReshapeTransformer()),
            #("Debugger",DebugTransformer()),
            #("scaler", StandardScaler()),
            ('classifier',self.learning)
            ])
        return pipeline
    
    def save_weights(self, filename):
        self.csp_model.save_model(filename)
    
    def get_dataset_train(self):
        return self.csp_model_train.get_dataset()

    def save_dataset_train(self, filename):
        self.csp_model_train.save_dataset(filename)
