import numpy as np
import json

def logistic_regression(X:np.array, y:np.array,accuracy=0.001, max_iter=10000, learning_rate=0.001, optimizer="gradient_descent", batch_size = 50):
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
	
	if optimizer != "gradient_descent":
		if optimizer == "stochastic_gradient_descent":
			x_batches = np.array_split(x, x.shape[0] / 1)
			y_batches = np.array_split(y, y.shape[0] / 1)
			batch_size = 1
		elif optimizer == "mini_batch_gradient_descent":
			x_batches = np.array_split(x, x.shape[0] / batch_size)
			y_batches = np.array_split(y, y.shape[0] / batch_size)
		else:
			raise ValueError("The optimizer must be 'gradient_descent', 'stochastic_gradient_descent' or 'mini_batch_gradient_descent'")
	else:
		x_batches = np.array([x])
		y_batches = np.array([y])
		batch_size = m

	# Initialize the parameters of the linear model
	theta = np.random.rand(x.shape[1],y.shape[1])
	
	# initialize the array of losses and predictions and the previous loss
	array_losses = np.empty(0)
	y_predicted = np.zeros(batch_size)
	prev_los = np.ones(y.shape[1]) * np.inf
	
	#start the loop to find the best parameters
	batch_nr = 0
	for _ in range(max_iter):
		for _ in x_batches:
			batch_size = x_batches[batch_nr].shape[0]
			y_predicted = np.exp(np.dot(x_batches[batch_nr], theta)) /np.exp(np.dot(x_batches[batch_nr], theta)).sum(axis = 1).reshape(batch_size,1)
			gradient = np.dot(x_batches[batch_nr].T, (y_predicted - y_batches[batch_nr]))
			theta = theta - gradient * learning_rate
			loss = (y_predicted - y_batches[batch_nr]).sum(axis = 0)
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

