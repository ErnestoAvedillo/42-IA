from sklearn.base import TransformerMixin, BaseEstimator

class DebugTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # no fitting necessary

    def transform(self, X):
        print("Data entering the classifier:")
        print("Shape:", X.shape)
        print("Sample (first 10 rows):", X[0:10])
        return X  # pass data through unchanged