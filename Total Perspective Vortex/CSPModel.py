import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef
)
from mne import io, Epochs, events_from_annotations
from mne.datasets import sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
#
# This is a class to handle the CSP model for EEG data
# It includes methods to fit the model, transform data, save and load the model,
# and plot various aspects of the model.
# The class uses the MNE library for EEG data processing and the sklearn library for machine learning.
# The CSP (Common Spatial Pattern) algorithm is used for feature extraction from EEG data.
# The class also includes methods to save and load the model parameters to and from a JSON file.
# The class is designed to be used in a pipeline for EEG data analysis and classification.
# The class is initialized with the number of components for the CSP model.
# The class includes methods to add data, fit the model, get and set weights,
# save and load the model, get the dataset, save the dataset, transform data,
# and plot various aspects of the model.
# The class also includes methods to plot the patterns, filters, scores, and weights of the CSP model.
# The class is designed to be flexible and can be used with different EEG datasets and classification tasks.
# The class is initialized with the number of components for the CSP model.
# The class includes methods to add data, fit the model, get and set weights,
# save and load the model, get the dataset, save the dataset, transform data,
# and plot various aspects of the model.
# The class also includes methods to plot the patterns, filters, scores, and weights of the CSP model.
# The class is designed to be flexible and can be used with different EEG datasets and classification tasks.

# Supongamos que df es tu DataFrame de dimensiones 29x5x670
# df = pd.read_csv('tu_archivo.csv')

# Crear una clase personalizada para el reshape
class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(X.shape[0], -1)

class CSPModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.covs = None
        self.MyCSPModel = CSP (n_components = n_components, reg = None, log = None, transform_into = "csp_space")
        self.labels = None
        self.W = None
        self.my_filename = None

    def add_data(self, data, label):
        if self.covs is None:
            self.covs = data
            self.labels = label
        else:
            self.covs = np.concatenate((self.covs, data), axis=0)
            self.labels = np.concatenate((self.labels,label))
        return self.covs, self.labels
    
    def fit(self,X,y):
        X_csp = self.MyCSPModel.fit_transform(X,y)
        print(f"caracterñisticas CPS {X_csp.shape}")
        print(f"caracterñisticas CPS { self.MyCSPModel.get_params()}")
        return X_csp

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

    """("estimator", GeneralizingEstimator(
                LinearModel(SVC(kernel='poly', C=1, gamma='scale', probability=True)),
                scoring="accuracy",
                n_jobs=1,
                verbose=True,
            ))
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
            """
    def make_pipeline(self, epochs_info):

        self.pipeline = Pipeline([
            ("csp",self.MyCSPModel),
            ('reshape',ReshapeTransformer()),
            ("scaler", StandardScaler()),
            ('classifier',SVC(kernel='poly', C=1, gamma='scale', probability=True))  
            ])
        return self.pipeline
    
    def fit_pipeline(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        print(self.pipeline.score(X_test, y_test))
    
    def predict_pipeline(self, X):
        return self.pipeline.predict(X)
    
    def set_param_pipeline(self, **params):
        self.pipeline.set_params(params)

    def get_param_pipeline(self):
        return self.pipeline.get_params()
    
    def score_pipeline(self, X, y):
        return self.pipeline.score(X, y)
    
    def cross_val_score_pipeline(self, X, y, cv=5):
        accuracy = cross_val_score(self.pipeline, X, y, cv=cv).mean()
        print(f"Cross-validation accuracy: {accuracy:.2f}")
        return accuracy
    
    def get_coef(self, X, y):
        return get_coef(self.pipeline, X, y)
    
    def plot_coef(self, X, y):
        return self.pipeline.plot_coef(X, y)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
