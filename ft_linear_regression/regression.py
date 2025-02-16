import numpy as np
"""
This is a simple linear regression algorithm that uses gradient descent to find the best parameters for a linear model. 
The algorithm receives two numpy arrays, x and y, an accuracy parameter that defaults to 0.01, and the maximum iterations to avoid being in an infinite loop. 
The function returns the slope and the intercept of the linear model that best fits the data.

The algorithm works as follows:
1. Check if the size of x and y is the same.
2. Initialize the parameters of the linear model to zero.
3. Reshape y to be a column vector.
4. Add a column of ones to the x matrix.
5. Calculate the predicted values of y using the current parameters.
6. Iterate until the maximum number of iterations is reached or the predicted values are close enough to the actual values.
7. Calculate the gradient of the cost function.
8. Update the parameters using the gradient and the learning rate.
9. Check if the predicted values are close enough to the actual values.
"""

def linear_regression(X:np.array, y:np.array,accuracy=0.1, max_iter=1000, learning_rate=0.01):
    if X.shape[0] != y.size:
        raise ValueError("x and y must have the same size")
    m = y.size
    y = y.reshape(m, 1)
    if X.ndim == 1:
        X = X.reshape(m, 1)
    # Calculate the vector to scale theta0
    array_losses = np.empty(10)
    factor_theta0 = np.concat((np.ones([1]),-(X.mean(axis = 0)/ X.std(axis = 0))), axis = 0)

    # Scale X to improve convergence

    x = (X - np.mean(X, axis = 0)) / (np.std(X, axis = 0))
    x = np.concat(( np.ones((m, 1)) ,x), axis = 1)
    theta = np.random.rand(x.shape[1],1)
    y_predicted = np.zeros(y.shape)
    prev_los = float('inf')
    for _ in range(max_iter):
        y_predicted = np.dot(x, theta)
        gradient = np.dot(x.T, (y_predicted - y)) / m
        theta = theta - gradient * learning_rate
        loss = np.pow(y_predicted - y, 2).sum() / m
        if abs(loss- prev_los) < accuracy:
            break
        prev_los = loss
        array_losses = np.concat([array_losses, np.array([loss])])
    #print_graph(x[:,1],y,theta)
    #print(f"antes del cambio{theta.T}")
    #print (f"matriz de cambio{np.std(X, axis = 0)}")
    theta = np.concat([np.dot(theta.T,factor_theta0), theta[1:,0] / np.std(X, axis = 0)])
    #print(f"después del cambio{theta}")
    return theta, array_losses
