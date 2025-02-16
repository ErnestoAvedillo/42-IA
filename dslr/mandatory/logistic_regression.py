import numpy as np
import json

def logistic_regression(X:np.array, y:np.array,accuracy=0.001, max_iter=10000, learning_rate=0.001):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same size")
    if np.isnan(X).any():
        raise ValueError("X array has NaN values.")
    if np.isnan(y).any():
        raise ValueError("y array has NaN values.")
    # Check if the input arrays are 1D and reshape them if necessary
    m = y.shape[0]
    if X.ndim == 1:
        X = X.reshape(m, 1)
    if y.ndim == 1:
        y = y.reshape(m, 1)
    
    # Calculate the vector to scale theta0
    factor_theta0 = np.concat((np.ones([1]),-(X.mean(axis = 0)/ X.std(axis = 0))), axis = 0)
    
    # Normalize the input data
    x = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
    
    # Add a column of ones to the input data
    x = np.concat(( np.ones((m, 1)) ,x), axis = 1)
    
    # Initialize the parameters of the linear model
    theta = np.random.rand(x.shape[1],y.shape[1])
    
    # initialize the array of losses and predictions and the previous loss
    array_losses = np.empty(0)
    y_predicted = np.zeros(y.shape)
    prev_los = np.ones(y.shape[1]) * np.inf
    
    #start the loop to find the best parameters
    for _ in range(max_iter):
        y_predicted = np.exp(np.dot(x, theta)) /np.exp(np.dot(x, theta)).sum(axis = 1).reshape(m,1)
        gradient = np.dot(x.T, (y_predicted - y))
        theta = theta - gradient * learning_rate
        loss = (y_predicted - y).sum(axis = 0)
        if np.max(abs(loss- prev_los)) < accuracy:
            break
        prev_los = loss
        array_losses = np.concat([array_losses, loss])
    
    # Scale the theta0 parameter
    theta = np.concat([np.dot(theta.T,factor_theta0).reshape(-1, 1).T, theta[1:,:] / np.std(X, axis = 0).reshape(-1, 1)],axis = 0)

    # Save the parameters in a json file
    arguments_file = "arguments.json"
    with open(arguments_file, "w", encoding="utf-8") as myfile:
        argument_dicc = {"arguments":theta.tolist()}
        json.dump(argument_dicc, myfile, indent = 4, ensure_ascii = False)
    return theta, array_losses    

