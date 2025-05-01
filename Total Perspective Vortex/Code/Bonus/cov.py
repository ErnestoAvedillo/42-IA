import numpy as np

def Normalize(X, y, ddof=0):
    """
    Normalize the covariance matrix

    :param X:
    :param y:
    :param ddof:ddof=1 will return the unbiased estimate, even if both fweights and aweights are specified
                ddof=0 will return the simple average
    :return:
    """
    clases, count = np.unique(y, return_counts=True)
    n_classes = len(clases)
    for i in range(n_classes):
        mask = y == clases[i]
        aux = X[mask].mean(axis=1)[:, np.newaxis]
        X [mask] = (X[mask] - aux) / (count[i] - ddof)
    return X, n_classes

def CalculateCovariance(X, y, ddof=0):

    X_normalized, n_classes = Normalize(X, y, ddof)
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    if n_classes >= 2:
        cov = np.einsum('ijk,ikl->ijl', X_normalized, X_normalized.transpose(0, 2, 1))
    else:
        raise ValueError("n_classes must be >= 2.")
    return cov