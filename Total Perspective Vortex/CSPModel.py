import numpy as np
import pandas as pd
import json

class CSPModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = {}
        for i in range(n_components):
            self.covs [i] = np.array([])
        self.W = None
        self.my_filename = None

    def add_data(self, data, label):
        cov = np.cov(data)
        if self.covs[label].size == 0:
            self.covs[label] = cov
        elif self.covs[label].ndim == 2:
            self.covs[label] = np.stack((self.covs[label], cov), axis=-1)
        else:
            self.covs[label] = np.concatenate((self.covs[label], np.expand_dims(cov, axis=-1)), axis=-1)

    def fit(self):
        covs_per_label = np.array([self.covs[label].mean(axis=2) for label in self.covs.keys()])
        system_cov = covs_per_label.sum(axis=0)
        eigenvalues, eigenventors = np.linalg.eigh(system_cov)
        #R = np.mean(class_covs, axis=0)
        #R_inv = np.linalg.pinv(R)
        #E, V = np.linalg.eigh(R_inv)
        #V = V[:, np.argsort(E)[::-1]]
        P = np.dot(np.diag(np.sqrt(1/eigenvalues)), eigenventors.T)
        S = covs_per_label.copy()
        print(len(S))
        for i in range(len(S)):
            S[i] = P @ system_cov @ P.T
        _, W_eigenvect = np.linalg.eigh(S)
        self.W = W_eigenvect[0]
        #All eigenvectors should be the same. Only the first is needed.
        return self.W

    def get_weights(self):
        return self.W
    
    def set_weights(self, W):
        self.W = W

    def save_model(self, filename):
        with open(filename, "w", encoding="utf-8") as myfile:
            weights_dicc = {"weights": self.W.tolist()}
            json.dump(weights_dicc, myfile, indent = 4, ensure_ascii = False)

    def load_model(self, filename):
        with open(filename, "r", encoding="utf-8") as myfile:
            data = json.load(myfile)
            self.W = data["houses"]

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

    def transform(self, X):
        return np.dot(self.W, X)