import numpy as np

class CSPModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = {}
        for i in range(n_components):
            self.covs [i] = np.array([])
        self.W = None

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
        self.W = np.dot(P, eigenventors)

    def transform(self, X):
        return np.dot(self.W, X)