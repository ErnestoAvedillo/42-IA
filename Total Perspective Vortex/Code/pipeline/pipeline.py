from .CSPModel import CSPModel
from mne.decoding import CSP, cross_val_multiscore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
class My_Pipeline(own_csp=True):
    def __init__(self, n_components = 4):
        self.n_components = n_components
        self.pipeline = None
        self.csp = None
        self.learning = None
        self.own_csp = own_csp

    def make_pipeline(self, classifier = None):
        self.learning = classifier
        #("csp",CSPModel (n_components = self.n_components)),
        if self.own_csp:
            self.csp = CSPModel (n_components = 4)
        else:    
            self.csp = CSP (n_components = 4, reg = None, log = None, transform_into = "average_power", rank = {'eeg':64}, norm_trace = False)
        #self.csp = CSP (n_components = 4, reg = None, log = True, norm_trace = False)
        self.pipeline = Pipeline([
            ("csp",self.csp),
            ("scaler", StandardScaler()),
            #("Debugger",DebugTransformer()),
            ('classifier',self.learning)
            ])

    def train_model(self,X_train,y_train):
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        self.scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, n_jobs=None)
        self.pipeline.fit(X_train, y_train)
        print(f"Pipeline fitted {self.scores}")
        return
    
    def test_model(self, X, y):
        y_pred = self.pipeline.predict(X)
        return self.evaluate_prediction(y, y_pred)
        
    
    def evaluate_prediction(self, Y, y_pred):
        print("Classification report:")
        print(classification_report(Y, y_pred, zero_division=0)) 
        print("Accuracy score:")
        print(accuracy_score(Y, y_pred))
        print("Precision, recall, f1-score:")
        precision, recall, f1_score, _ = precision_recall_fscore_support(Y, y_pred, average='weighted')
        results = [precision, recall, f1_score]
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return results

