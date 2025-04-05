import numpy as np
import pandas as pd
import os
from mne import  channels, io, events_from_annotations, Epochs, time_frequency
from .event_type import Event_Type
from CSPModel import CSPModel

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
    
    def fill_model(self, mask) -> CSPModel:
        data_raws = []
        for file, is_train in zip(self.files, mask):
            if not is_train:
                continue
            my_raw = io.read_raw_edf(file, preload=True, verbose = "error")
            event_list = self.get_events(my_raw)
            if event_list is None or event_list.size == 0:
                continue
            data_raws.append(my_raw)
        self.concat_raw(data_raws)
        tmin = -0.2  # 200 ms before the event
        tmax = 4   # 500 ms after the event
        self.epochs = Epochs(self.raw, self.events, self.event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
        spectrum= self.raw.compute_psd(method="welch", fmin=4, fmax=40)
        self.csp_model = CSPModel(self.n_components)
        self.X = self.epochs.get_data()  # (n_trials, n_channels, n_samples)
        self.Y = self.epochs.events[:, -1]  # Etiquetas
        self.csp_model.add_data(self.X, self.Y)
        return self.csp_model

    def train_model(self):
        self.csp_model_train = self.fill_model( self.mask)
        self.csp_model_train.make_pipeline(self.epochs.info)
        self.csp_model_train.fit_pipeline(self.X, self.Y)
        self.csp_model_train.plot_patterns(self.epochs.info)
        self.csp_model_train.save_model("csp_model.json")

    def test_model(self):
        parameters =self.csp_model_train.get_param_pipeline()
        if parameters is None:
            print("Train model first.")
            return
        self.csp_model_test = self.fill_model( ~self.mask)
        self.csp_model_test.set_param_pipeline(parameters)
        self.csp_model_test.set_weights(self.csp_model_train.get_weights())
        self.csp_model_test.transform(self.csp_model_train.covs, self.csp_model_train.labels)
    
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
