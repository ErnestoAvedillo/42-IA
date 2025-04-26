def get_classifiers_list():
    classifiers={
                "KNN":               "K-Nearest Neighbors",
                "KNN_O_Vs_O":        "OneVsOneClassifier K-Nearest Neighbors",
                "KNN_O_Vs_R":        "OneVsRestClassifier K-Nearest Neighbors",
                "DT":                "DecisionTree",
                "DT_O_Vs_O":         "OneVsOneClassifier DecisionTree",
                "DT_O_Vs_R":         "OneVsRestClassifier DecisionTree",
                "SVM":               "K-Nearest Neighbors",
                "SVM_O_Vs_O":        "OneVsOneClassifier K-Nearest Neighbors",
                "SVM_O_Vs_R":        "OneVsRestClassifier K-Nearest Neighbors",
                "POLY_SVM":          "Suport Vector Machine (Linear)",
                "POLY_SVM_O_Vs_O":   "OneVsOneClassifier Suport Vector Machine (Linear)",
                "POLY_SVM_O_Vs_R":   "OneVsRestlassifier Suport Vector Machine (Linear)",
                "SIG_SVM":           "Suport Vector Machine (Polinomial)",
                "SIG_SVM_O_Vs_O":    "OneVsOneClassifier Suport Vector Machine (Polinomial)",
                "SIG_SVM_O_Vs_R":    "OneVsRestlassifier Suport Vector Machine (Polinomial)",
                "RF":                "Suport Vector Machine (Sigmoid)",
                "RF_O_Vs_O":         "OneVsOneClassifier Suport Vector Machine (Sigmoid)",
                "RF_O_Vs_R":         "OneVsRestlassifier Suport Vector Machine (Sigmoid)",
                "LDA":               "Random Forest",
                "LDA_O_Vs_O":        "OneVsOneClassifier Random Forest",
                "LDA_O_Vs_R":        "OneVsRestlassifier Random Forest",
                "LR":                "Logistic Regression",
                "LR_O_Vs_O":         "OneVsOneClassifier Logistic Regression",
                "LR_O_Vs_R":         "OneVsRestlassifier Logistic Regression",
    }
    return classifiers

def msg_error():
    print("Arguments to execute this script are:")
    print('pyhton Mandatory/mybci.py "<subjects>" "<runs>" "<classifier>"')
    print("Example:")
    print('pyhton Mandatory/mybci.py "[1,2]" "[4,8,12]" "PLD"')
    print("subjects can be from 1 to 109")
    print("runs can be from 3 to 14")
    print("classifiers are:")
    classifiers = get_classifiers_list()
    for item, val in classifiers.items():
        print(f"{item}: to implement {val}")