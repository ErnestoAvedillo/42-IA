import pandas as pd
import numpy as np
from logistic_regression import logistic_regression
from logistic_prediction import predict
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

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


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Describe the CSV to be trained.")
	parser.add_argument("-f", "--file", type=str, help="Path to the CSV file to train.")
	parser.add_argument("-a","--args", type=str, help="Path to the JSON file with the arguments.")
	args = parser.parse_args()

	if not args.file:
		print("Please give a file as argument to train.")
		print("Example: python histogram.py --file hogwarts.csv")
		print("For help:python histogram.py -h or python histogram.py --help")
		exit(1)
	if not args.args:
		arguments_file = "arguments.json"
	try:
		df = pd.read_csv(args.file)
		arguments_file = args.args
	except:
		print("The file you entered does not exist or you don't have access.")
		exit(1)


df = df.dropna(axis=1, how = 'all')
df = df.dropna(axis = 0)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train the model with the training data
X,Y,options = obtain_data(train_data)
epochs = 1
for i in range(epochs):
	print ("Epoch nr: ", i)
	theta, _ = logistic_regression(X,Y,optimizer="mini_batch_gradient_descent",batch_size = 50)

	#test ethe model with the test data
	X_test,_, _ = obtain_data(test_data, options)

	predictions = predict(X_test, theta,options)

	Y_test = test_data['Hogwarts House'].to_numpy()

	#review the acuracy, confusion matrix and clasification report of the predictons

	# Accuracy
	accuracy = accuracy_score(Y_test, predictions)
	print("Accuracy:", accuracy)

	# Confusion Matrix
	conf_matrix = confusion_matrix(Y_test, predictions)
	print("Confusion Matrix:\n", conf_matrix)

	# Classification Report (Precision, Recall, F1-score)
	report = classification_report(Y_test, predictions, zero_division=0)
	print("Classification Report:\n", report)

# Once everithing seems to be working, train the model with the whole dataset

X,Y,_ = obtain_data(df, options)
theta, loss = logistic_regression(X,Y,optimizer="mini_batch_gradient_descent")

# Save the arguments and the options in a json file
with open(arguments_file, "w", encoding="utf-8") as myfile:
	argument_dicc = {"arguments":theta.tolist(), "houses":options.tolist()}
	json.dump(argument_dicc, myfile, indent = 4, ensure_ascii = False)
