
import numpy as np

class PCAModel:
    """
    Principal component analysis (PCA)
    
    Parameters
    ----------
    n_components : int
        top number of principal components to keep
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # standardize
        X = X.copy()
        self.mean = np.mean(X, axis = 0)
        self.scale = np.std(X, axis = 0)
        X = (X - self.mean) / self.scale
        
        # eigendecomposition
        eig_vals, eig_vecs = np.linalg.eig(np.cov(X.T))
        self.components = eig_vecs[:, :self.n_components]
        var_exp = eig_vals / np.sum(eig_vals)
        self.explained_variance_ratio = var_exp[:self.n_components]
        return self

    def transform(self, X):
        X = X.copy()
        X = (X - self.mean) / self.scale
        X_pca = X @ self.components
        return X_pca
    