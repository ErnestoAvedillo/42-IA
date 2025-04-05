import numpy as np
import pandas as pd
import json
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
class CSPModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = None
        self.MyCSPModel = CSP (n_components = n_components, reg = None, log = None, transform_into = "csp_space")
        self.labels = None
        self.W = None
        self.my_filename = None

    def fit(self,X,y):
        self.MyCSPModel.fit(X,y)
        return self
    
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
