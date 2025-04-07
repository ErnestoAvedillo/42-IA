import numpy as np
import pandas as pd
import json
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from mne import channels, io, events_from_annotations, Epochs
from pipeline.event_type import EventTypes
class CSPModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = np.array([])
        self.labels = np.array([])
        self.W = []
        self.my_filename = None
        self.covs_averaged = None

    def load_events(self, files):
        labels = []
        covs = []
        for file in files:
            raw = io.read_raw_edf(file, preload=True)
            tmin = -0.2  # 200 ms before the event
            tmax = 4   # 500 ms after the event
            events, event_id = events_from_annotations(raw)
            epochs = Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = "error")
            X = epochs.get_data()
            Y = epochs.events[:, -1]
            cov =  np.einsum('ijk,ikl->ijl', X, X.transpose(0, 2, 1))
            traces = np.sum(np.diagonal(cov, axis1=1, axis2=2), axis=1)
            trace_reshaped = traces[:,np.newaxis, np.newaxis]
            covs.append(cov / trace_reshaped)
            event_type = EventTypes(filename = file)
            labels.append(event_type.convert_event_labels(Y))
        self.covs = np.vstack(covs)
        self.labels = np.stack(labels).flatten()
        return 
    
    def fill(self, X:np.array, Y:np.array):
        covs_averaged = np.zeros(self.n_components)
        cov_matrices = np.einsum('ijk,ikl->ijl', X, X.transpose(0, 2, 1))

    
    def fit(self):
        events = np.unique(self.labels)
        averaged =[]
        for event in events:
            mask = self.labels == event
            masked_covs = self.covs[mask].mean(axis = 0)
            averaged.append(masked_covs)
        covs_averaged = np.array(averaged)
        global_cov = covs_averaged.sum(axis = 0)
        eigenvalues_arr = []
        eigenvectors_arr = []
        for i in range(covs_averaged.shape[0]):
            eigenvalues, eigenvectors = np.linalg.eigh(covs_averaged[i],global_cov)
            eigenvalues_arr.append(eigenvalues)
            eigenvectors_arr.append(eigenvectors)
        self.eigenvalues = np.array(eigenvalues_arr)
        self.eigenvectors = np.array(eigenvectors_arr)
        
        global_eigenvalues, global_eigenvectors = np.linalg.eigh(global_cov)
        P = np.divide(global_eigenvectors, np.sqrt(global_eigenvalues))
        B = P @ global_eigenvectors.T
        return
    
    def transform(self, data):
        self.MyCSPModel.transform(data)
        return self

    def get_weights(self):
        return self.MyCSPModel.get_params()
    
    def set_weights(self, params):
        self.MyCSPModel.set_params(params)

    def save_model(self, filename):
        with open(filename, "w", encoding="utf-8") as myfile:
            weights_dicc = {"weights": self.get_weights().tolist()}
            json.dump(weights_dicc, myfile, indent = 4, ensure_ascii = False)

    def load_model(self, filename):
        with open(filename, "r", encoding="utf-8") as myfile:
            data = json.load(myfile)
            self.MyCSPModel.set_params(data)

    def get_dataset(self):
        data = self.covs[0]
        Y = np.zeros(data.shape[2])
        for i in range (1, self.n_components):
            if self.covs[i].ndim == 1:
                print(f"Error in label {i} with shape {self.covs[i].shape}")
                continue
            elif self.covs[i].ndim == 2:
                self.covs[i] = np.expand_dims(self.covs[i], axis = 2)
            data = np.concatenate((data, self.covs[i]), axis = 2)
            Y = np.concatenate((Y, np.ones(self.covs[i].shape[2]) * i), axis = 0)
        data = np.dot(self.W, data)
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        Y = Y.reshape(Y.shape[0], 1)
        data = np.concatenate((data.T, Y), axis = 1)
        return data
    
    def save_dataset(self, filename):
        self.my_filename = filename
        data = self.get_dataset()
        np.random.shuffle(data)
        pd.DataFrame(data).to_csv(filename, header = False, index = False)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
