import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import  channels, io, events_from_annotations, Epochs, time_frequency, Annotations, concatenate_raws
from mne.preprocessing import ICA, create_eog_epochs
from .event_type import Event_Type
from .classifiers import Classifier
from .CSPModel import CSPModel
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    PSDEstimator
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from mne.datasets import sample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from .reshape_transformer import ReshapeTransformer
from .Debugger import DebugTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
class My_Pipeline():
    def __init__(self, n_components = 4):
        self.weights = None
        self.n_components = n_components
        self.pipeline = None
        self.csp = None
        self.mycsp = None
        self.learning = None

    def make_pipeline(self, classifier = None):
        self.learning = classifier
        #("csp",CSPModel (n_components = self.n_components)),
        #self.csp = CSP (n_components = 4, reg = None, log = None, transform_into = "average_power", rank = {'eeg':64}, norm_trace = False)
        self.csp = CSP (n_components = 4, reg = None, log = True, norm_trace = False)
        self.mycsp = CSPModel (n_components = 4)
        self.pipeline = Pipeline([
            ("csp",self.mycsp),
            #('reshape',ReshapeTransformer()),
            #("Debugger",DebugTransformer()),
            #("scaler", StandardScaler()),
            ('classifier',self.learning)
            ])

    def train_model(self,X,y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        #self.scores = cross_val_multiscore(self.pipeline, X_train, y_train, cv=cv, n_jobs=None)
        self.scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, n_jobs=None)
        print(f"Scores are {self.scores}")
        self.pipeline.fit(X_train, y_train)
        return self.test_model(X_val,y_val)

    def test_model(self, X, y):
        y_pred = self.pipeline.predict(X)
        return self.evaluate_prediction(y, y_pred)
        
    
    def evaluate_prediction(self, Y, y_pred):
        print("Classification report:")
        print(classification_report(Y, y_pred)) 
        print("Accuracy score:")
        print(accuracy_score(Y, y_pred))
        print("Precision, recall, f1-score:")
        precision, recall, f1_score, _ = precision_recall_fscore_support(Y, y_pred, average='weighted')
        results = [precision, recall, f1_score]
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return results

    def get_weights(self):
        pipeline_params = {}
        for key, value in self.pipeline.get_params().items():
            pipeline_params[key] = value
        return pipeline_params
    
    def set_weights(self, **params):
        for key, value in params.items():
            value.set_params(**{key: value})
        return self.pipeline
