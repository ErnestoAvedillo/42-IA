import numpy as np
import pandas as pd
import json
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from mne import channels, io, events_from_annotations, Epochs
from pipeline.event_type import Event_Type
class CSPModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = np.array([])
        self.labels = np.array([])
        self.W = []
        self.my_filename = None
        self.covs_averaged = None

    def _calc_covariance(self, X, ddof=0):
        """
        Calculate the covariance based on numpy implementation

        :param X:
        :param ddof:ddof=1 will return the unbiased estimate, even if both fweights and aweights are specified
                    ddof=0 will return the simple average
        :return:
        """
        X -= X.mean(axis=1)[:, None]
        N = X.shape[1]
        return np.dot(X, X.T.conj()) / float(N - ddof)
    
    def fit(self, X, Y):
        cov =  np.einsum('ijk,ikl->ijl', X, X.transpose(0, 2, 1))
        traces = np.sum(np.diagonal(cov, axis1=1, axis2=2), axis=1)
        trace_reshaped = traces[:,np.newaxis, np.newaxis]
        cov_matrices = cov / trace_reshaped
        events = np.unique(Y)
        averaged =[]
        for event in events:
            mask = Y == event
            masked_covs = cov_matrices[mask].mean(axis = 0)
            averaged.append(masked_covs)
        covs_averaged = np.array(averaged)
        global_cov_averaged = covs_averaged.sum(axis = 0)
        global_eigenvalues, global_eigenvectors = np.linalg.eigh(global_cov_averaged)
        P = np.divide(global_eigenvectors, np.sqrt(global_eigenvalues))
        self.W = P @ global_cov_averaged @ P.T
        eigenvalues_arr = []
        eigenvectors_arr = []
        for i in range(covs_averaged.shape[0]):
            
            eigenvalues, eigenvectors = np.linalg.eigh(global_cov_averaged)
            eigenvalues_arr.append(eigenvalues)
            eigenvectors_arr.append(eigenvectors)
        self.eigenvalues = np.array(eigenvalues_arr)
        self.eigenvectors = np.array(eigenvectors_arr)
        
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
