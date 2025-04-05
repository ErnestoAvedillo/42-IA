from sklearn.base import BaseEstimator, TransformerMixin
class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(X.shape[0], -1)