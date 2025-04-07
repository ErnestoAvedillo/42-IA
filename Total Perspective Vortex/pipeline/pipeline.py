import numpy as np
import pandas as pd
import os
from mne import  channels, io, events_from_annotations, Epochs, time_frequency
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

    def open_raw(self, file):
        self.raw = io.read_raw_edf(file,preload=True)
        self.raw.rename_channels({old: new for old, new in zip(self.raw.ch_names, self.ch_names)})
        self.raw.set_montage(self.montage)
        self.raw = self.raw.filter(l_freq = self.lfreq, h_freq = self.hfreq , fir_design='firwin')
        # Seleccionar solo los canales de EEG
        self.raw.pick_types(eeg=True)

    def get_events(self):
        self.events, self.event_id = events_from_annotations(self.raw)
        self.event_names = {v: k for k, v in self.event_id.items()}  # Map IDs to event names


    def define_test_train(self, percentage = None, mask = None):
        if mask is None and percentage is None:
            percentage = 0.8
        if mask is not None:
            self.mask = mask
            return
        self.mask = np.array(np.random.rand(len(self.files)) < percentage)
    
    def fill_model(self, mask) -> CSPModel:
        for file, is_train in zip(self.files, mask):
            if not is_train:
                continue
            self.open_raw(file)
            self.get_events()
            if self.events is None or len(self.events) == 0:
                continue
            self.files.append(file)
        return
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
    def train_model(self):
        self.csp_model_train = CSPModel(self.n_components)
        self.fill_model( self.mask)
        self.csp_model_train.load_events(self.files)
        self.csp_model_train.fit()
        self.csp_model_train.fit_pipeline(self.X, self.Y)
        self.csp_model_train.plot_patterns(self.ch_names)
        self.csp_model_train.plot_filters()
        self.csp_model_train.plot_scores()
        self.csp_model_train.plot_weights()
        self.csp_model_train.save_model("csp_model.json")

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
