from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class Classifier:
    def __init__(self, classifier = None):
        """
        Initializes the Classifier with a specific type.
        
        Args:
            classifier_type (str): The type of classifier to be used.
        """
        self.dict_classifiers = {
            "KNN":               KNeighborsClassifier(n_neighbors=5),
            "KNN_O_Vs_O":        OneVsOneClassifier(KNeighborsClassifier(n_neighbors=5)),
            "KNN_O_Vs_R":        OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
            "DT":                DecisionTreeClassifier(random_state=42),
            "DT_O_Vs_O":         OneVsOneClassifier(DecisionTreeClassifier(random_state=42)),
            "DT_O_Vs_R":         OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
            "SVM":               SVC(kernel='linear', C=1, gamma='scale', probability=True),
            "SVM_O_Vs_O":        OneVsOneClassifier(SVC(kernel='linear', C=1, gamma='scale', probability=True)),
            "SVM_O_Vs_R":        OneVsRestClassifier(SVC(kernel='linear', C=1, gamma='scale', probability=True)),
            "POLY_SVM":          OneVsOneClassifier(SVC(kernel='poly', C=1, degree=3, probability=True)),
            "POLY_SVM_O_Vs_O":   OneVsOneClassifier(SVC(kernel='poly', C=1, degree=3, probability=True)),
            "POLY_SVM_O_Vs_R":   OneVsRestClassifier(SVC(kernel='poly', C=1, degree=3, probability=True)),
            "SIG_SVM":           SVC(kernel='sigmoid', C=0.1, gamma=0.5, probability=True),
            "SIG_SVM_O_Vs_O":    OneVsOneClassifier(SVC(kernel='sigmoid', C=0.1, gamma=0.5, probability=True)),
            "SIG_SVM_O_Vs_R":    OneVsRestClassifier(SVC(kernel='sigmoid', C=0.1, gamma='scale', probability=True)),
            "RF":                OneVsOneClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
            "RF_O_Vs_O":         OneVsOneClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
            "RF_O_Vs_R":         OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
            "LDA":               LinearDiscriminantAnalysis(),
            "LDA_O_Vs_O":        OneVsOneClassifier(LinearDiscriminantAnalysis()),
            "LDA_O_Vs_R":        OneVsRestClassifier(LinearDiscriminantAnalysis()),
            "LR":                LogisticRegression(solver="liblinear"),
            "LR_O_Vs_O":         OneVsOneClassifier(LogisticRegression(solver="liblinear")),
            "LR_O_Vs_R":         OneVsRestClassifier(LogisticRegression(solver="liblinear"))
            }
#        params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#            "": KernelPCA(n_components=5, kernel='rbf', gamma=0.5)
#            "": GridSearchCV(SVC(kernel='rbf', gamma=0.5, C=0.1), params_grid, cv=9)
 
    def get_dict_keys(self):
        return self.dict_classifiers.keys()
    
    def get_classifier_type(self, argument):
        """
        Returns the classifier type based on the incoming argument.
        
        Args:
            argument (str): The input argument to determine the classifier type.
        
        Returns:
            str: The classifier type.
        """
        return self.classifier_key
    
    def get_classifier(self):
        """
        Sets the classifier type
        
        Args:
            None

        Returns the classifier class
        """
        return self.classifier_type
    
    def set_classifier(self, classifier):
        """
        Sets the classifier type
        
        Args:
            argument (str): The input argument to determine the classifier type.

        Returns the classifier class
        """
        if classifier not in self.dict_classifiers or classifier is None:
            return None
        
        self.classifier_key = classifier
        self.classifier_type = self.dict_classifiers[classifier]
        return self.classifier_type