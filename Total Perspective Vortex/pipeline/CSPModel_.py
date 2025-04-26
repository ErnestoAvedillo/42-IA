""" Reimplementation of a signal decomposition using the Common spatial pattern"""

# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CSPModel(TransformerMixin, BaseEstimator):
    """
    CSP implementation based on MNE implementation

    https://github.com/mne-tools/mne-python/blob/f87be3000ce333ff9ccfddc45b47a2da7d92d69c/mne/decoding/csp.py#L565
    """
    def __init__(self, n_components=4, log=None, transform_into='average_power'):
        """
        Initializing the different optional parameters.
        Some checks might not be full, and all options not implemented.
        We just created the parser based on the original implementation of the CSP of MNE.

        :param n_components:
        :param log:
        :param transform_into:
        """
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components

        self.log = log

        self.transform_into = transform_into
        self._classes = 0
        self.filters_ = None
        self.mean_ = 0
        self.std_ = 0

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

    def _compute_covariance_matrices(self, X, y):
        """
        Compute covariance to every class

        :param X:ndarray, shape (n_epochs, n_channels, n_times)
                The data on which to estimate the CSP.
        :param y:array, shape (n_epochs,)
                The class for each epoch.

        :return:instance of CSP
        """
        _, n_channels, _ = X.shape
        covs = []

        for this_class in self._classes:
            x_class = X[y == this_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            # calc covar matrix for class
            covar_matrix = self._calc_covariance(x_class)
            is_hermitian = np.allclose(covar_matrix, covar_matrix.conj().T)
            covs.append(covar_matrix)
        return np.stack(covs)

    def fit(self, X, y):
        """
        Estimate the CSP decomposition on epochs.

        :param X:ndarray, shape (n_epochs, n_channels, n_times)
                The data on which to estimate the CSP.
        :param y:array, shape (n_epochs,)
                The class for each epoch.

        :return:instance of CSP
        """
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        else:
            covs = self._compute_covariance_matrices(X, y)
            eigen_vectors, eigen_values = self._decompose_covs(covs)
            ix = self._order_components(eigen_values)
            eigen_vectors = eigen_vectors[:, ix]
            self.filters_ = eigen_vectors.T
            pick_filters = self.filters_[:self.n_components]

            X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
            X = (X ** 2).mean(axis=2)

            # Standardize features
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)

            return self

    def transform(self, X):
        """
        Estimate epochs sources given the CSP filters.

        :param X: ndarray
        :return: ndarray
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X

    def fit_transform(self, X, y):
        """
        Apply fit and transform

        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        self.fit(X, y)
        return self.transform(X)

    def _decompose_covs(self, covs):
        """
         Return the eigenvalues and eigenvectors of a complex Hermitian ( conjugate symmetric )

        :param covs:
        :return:
        """
        from scipy import linalg
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise Exception("Not Handled")
        return eigen_vectors, eigen_values

    def _order_components(self, eigen_values):
        """
        Sort components using the mutual info method.

        :param eigen_values:
        :return:
        """
        n_classes = len(self._classes)
        if n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            raise Exception("Not Handled")
        return ix
    
"""    def get_model_params(self, deep = True):
        csp_params = {}
        csp_params['filters_'] = self.filters_
        csp_params['n_components'] = self.n_components
        csp_params['log'] = self.log
        csp_params['transform_into'] = self.transform_into
        csp_params['mean_'] = self.mean_
        csp_params['std_'] = self.std_
        csp_params['classes_'] = self._classes
        params = {}
        params['super'] = super().get_params(deep)
        params['csp_params'] = csp_params
        return params
    
    def set_model_params(self, **params):
        csp_params = params.pop('csp_params', {})
        self.filters_ = csp_params.get('filters_', None)
        self.n_components = csp_params.get('n_components', 4)
        self.log = csp_params.get('log', None)
        self.transform_into = csp_params.get('transform_into', 'average_power')
        self.mean_ = csp_params.get('mean_', 0)
        self.std_ = csp_params.get('std_', 1)
        self._classes = csp_params.get('classes_', 0)

        # Call the parent class's set_params method
        # to ensure that any other parameters are set correctly
        super().set_params(**params['super'])
        return """