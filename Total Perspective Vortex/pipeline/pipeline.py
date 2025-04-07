import numpy as np
import pandas as pd
import os
from mne import  channels, io, events_from_annotations, Epochs, annotations_from_events, Annotations
from .event_type import Event_Type
from CSPModel import CSPModel
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
from mne import io, Epochs, events_from_annotations
from mne.datasets import sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from reshape_transformer import ReshapeTransformer
from Debugger import DebugTransformer
class pipeline():
    def __init__(self, folder = None, filename=None):
        self.weights = None
        self.files = []
        self.ch_names = None
        self.montage = None
        self.lfreq = 0.1
        self.hfreq = 40
        self.mask = None
        self.events = None
        self.event_id = None
        self.event_names = None
        self.csp_model_test = None
        self.csp_model_train = None
        self.n_components = None
        self.raw  = None
        self.pipeline = None
        self.X = None
        self.Y = None
        if folder is not None:
            self.add_files_from_folder(folder)
        if filename is None:
            self.add_file(filename)

    def add_files_from_folder(self, folder = None):
        for ruta, carpeta, archivos in os.walk(folder):
            for archivo in archivos:
                self.add_file(os.path.join(ruta, archivo))
    
    def add_file(self, filename = None):
        if filename is None:
            return
        if os.path.exists(filename) and filename.endswith(".edf"):
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

    def concat_raw(self, raws):
        self.raw = io.concatenate_raws(raws)
        self.raw.rename_channels({old: new for old, new in zip(self.raw.ch_names, self.ch_names)})
        self.raw.set_montage(self.montage)
        self.raw = self.raw.filter(l_freq = self.lfreq, h_freq = self.hfreq , fir_design='firwin')
        # Seleccionar solo los canales de EEG
        self.raw.pick_types(eeg=True)

    def get_events(self,raw = None):
        if raw is None:
            raw = self.raw
        if raw is None:
            print("No raw data available.")
            return None
        self.events, self.event_id = events_from_annotations(raw)
        self.event_names = {v: k for k, v in self.event_id.items()}  # Map IDs to event names
        return self.events

    def define_test_train(self, percentage = None, mask = None):
        if mask is None and percentage is None:
            percentage = 0.8
        if mask is not None:
            self.mask = mask
            return
        self.mask = np.array(np.random.rand(len(self.files)) < percentage)
    
    def fill_model(self, mask):
        data_raws = []
        filenames = ["R05.edf","R06.edf","R09.edf","R10.edf","R13.edf","R14.edf"]
        # Read the files and create a list of raw objects
        # Iterate over the files and load them
        for file, is_train in zip(self.files, mask):
            if not is_train:
                continue
            my_raw = io.read_raw_edf(file, preload=True, verbose = "error")
            # Get events from annotations
            events, event_id = events_from_annotations(my_raw)

            data_raws.append([file,my_raw,events, event_id])
        data_X = []
        data_Y = []
        for file, my_raw, events, event_id in data_raws:
            # Get events from annotations
            events, event_id = events_from_annotations(my_raw)
            # Convert events to the correct format
            tmin = -0.2  # 200 ms before the event
            tmax = 4   # 500 ms after the event ---- 
            epochs = Epochs(my_raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
            if epochs is None or len(epochs) == 0:
                continue
            X = epochs.get_data()  # (n_trials, n_channels, n_samples)
            labels = epochs.events[:, -1]  # Etiquetas
            if any(word in file for word in filenames):
                labels[labels == 2] = 4
                labels[labels == 3] = 5
            data_X.append(epochs.get_data())
            data_Y.append(labels)
        self.X = np.concatenate(data_X, axis=0)
        self.Y = np.concatenate(data_Y, axis=0)
        return 

    def train_model(self):
        self.fill_model( self.mask)
        self.pipeline = self.make_pipeline()
        self.pipeline.fit(self.X, self.Y)

    def test_model(self):
        self.fill_model( ~self.mask)
        y_pred = self.pipeline.predict(self.X)
        print(f"score: {self.pipeline.score(self.X, self.Y)}")
        print(f"accuracy: {np.mean(y_pred == self.Y)}")
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

        #("csp",CSPModel (n_components = self.n_components)),
        self.csp = CSP (n_components = 10, reg = None, log = None, transform_into = "average_power", rank = {'eeg':64}, norm_trace = False)
        self.learning = SVC(kernel='poly', C=1, gamma='scale', probability=True)
        pipeline = Pipeline([
            ("csp",self.csp),
            #('reshape',ReshapeTransformer()),
            #("scaler", StandardScaler()),
            #('classifier', LinearDiscriminantAnalysis())
            ("debug", DebugTransformer()),
            ('classifier',self.learning),  
            #('classifier',RandomForestClassifier(n_estimators=100, random_state=42))  
            ])
        return pipeline
    
    def fit_pipeline(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        print(self.pipeline.score(X_test, y_test))
    
    def predict_pipeline(self, X):
        return self.pipeline.predict(X)
    
    def set_param_pipeline(self, **params):
        self.pipeline.set_params(params)

    def get_param_pipeline(self):
        return self.pipeline.get_params()
    
    def score_pipeline(self, X, y):
        return self.pipeline.score(X, y)
    
    def cross_val_score_pipeline(self, X, y, cv=5):
        accuracy = cross_val_score(self.pipeline, X, y, cv=cv).mean()
        print(f"Cross-validation accuracy: {accuracy:.2f}")
        return accuracy


    def save_weights(self, filename):
        self.csp_model.save_model(filename)
    
    def get_dataset_test(self):
        return self.csp_model_test.get_dataset()

    def save_dataset_test(self, filename):
        self.csp_model_test.save_dataset(filename)
    
    def get_dataset_train(self):
        return self.csp_model_train.get_dataset()

    def save_dataset_train(self, filename):
        self.csp_model_train.save_dataset(filename)
