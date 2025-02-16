import pandas as pd
import numpy as np
from logistic_regression import logistic_regression
from logistic_prediction import predict
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def obtain_data(data:pd.DataFrame, options = None):
	X = data[['Astronomy', 
		'Herbology', 
		'Defense Against the Dark Arts', 
		'Divination', 
		'Muggle Studies',
		'Ancient Runes', 
		'History of Magic', 
		'Transfiguration', 
		'Potions',
		'Charms', 
		'Flying']].to_numpy()
	if options is None:
		options = df['Hogwarts House'].unique()
	Y = np.array([data['Hogwarts House'] == option for option in options]).T.astype(int)
	return X,Y, options


df = pd.read_csv("../datasets/dataset_train.csv")
df = df.dropna(axis=1, how = 'all')
df = df.dropna(axis = 0)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train the model with the training data
X,Y,options = obtain_data(train_data)

theta, _ = logistic_regression(X,Y,optimizer="stochastic_gradient_descent")

#test ethe model with the test data
X,Y, _ = obtain_data(test_data, options)

predictions = predict(X, theta,options)

Y = test_data['Hogwarts House'].to_numpy()

#review the acuracy, confusion matrix and clasification report of the predictons

# Accuracy
accuracy = accuracy_score(Y, predictions)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(Y, predictions)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report (Precision, Recall, F1-score)
report = classification_report(Y, predictions)
print("Classification Report:\n", report)


# Once everithing seems to be working, train the model with the whole dataset

X,Y,_ = obtain_data(df, options)
theta, loss = logistic_regression(X,Y,optimizer="stochastic_gradient_descent")

# Save the arguments and the options in a json file
arguments_file = "arguments.json"
houses_file = "houses.json"
with open(arguments_file, "w", encoding="utf-8") as myfile:
	argument_dicc = {"arguments":theta.tolist()}
	json.dump(argument_dicc, myfile, indent = 4, ensure_ascii = False)

with open(houses_file, "w", encoding="utf-8") as myfile:
	argument_dicc = {"houses":options.tolist()}
	json.dump(argument_dicc, myfile, indent = 4, ensure_ascii = False)