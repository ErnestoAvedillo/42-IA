import pandas as pd
import numpy as np

def predict(data:np.array, theta:np.array, options = None):
    if options is None:
        print("Please provide the options for the houses")
        return None
    
    #add the bias term
    m = data.shape[0]
    data = np.concat(( np.ones((m, 1)) ,data), axis = 1)
    
    #predict the house
    y_predicted = np.exp(np.dot(data, theta)) /np.exp(np.dot(data, theta)).sum(axis = 1).reshape(m,1)
    y_predicted = np.argmax(y_predicted, axis = 1)

    #vonvert the predictions to the house names
    predictions = []
    for house in y_predicted:
        predictions.append(options[house])
    return np.array(predictions)
