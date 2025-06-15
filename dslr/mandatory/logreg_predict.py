import pandas as pd
import sys
import numpy as np
import json
from logistic_prediction import predict
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="The CSV file data to predict.")
	parser.add_argument("-f", "--file", type=str, help="Path to the CSV file to predict.")
	parser.add_argument("-a","--args", type=str, help="Path to the JSON file with the arguments.")
	args = parser.parse_args()

	if not args.file or not args.args:
		print("Please give a file as argument to predict and the arguments to use.")
		print("Example: python logreg_predict.py --file dataset_test.csv --args arguments.json")
		print("For help:python logreg_predict.py -h or python logreg_predict.py --help")
		exit(1)
	try:
		df = pd.read_csv(args.file)
		data_file = args.args
		#get the arguments and the options for the houses
		with open(data_file, "r", encoding="utf-8") as my_file:
			data = json.load(my_file)
			theta = np.array(data["arguments"])
			options = data["houses"]
	except:
		print("The dataset file or arguments file you entered does not exist or you don't have access.")
		exit(1)


# fill missing values with the mean of the column
df = df.fillna(df.mean(numeric_only=True))

#extract the features
X = df[['Astronomy', 
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

# run the predictions
try:
	list_of_houses = predict(X, theta,options)
except Exception as e:
	print(f"An error occurred during prediction: {e}")
	sys.exit(1)

#save the predictions in a csv file
df_houses = pd.DataFrame(list_of_houses, columns = ["Hogwarts House"])
df_houses.to_csv("houses.csv", index = True, index_label="Index")

df_houses_truth = pd.read_csv("dataset_truth.csv")
df_houses_truth = df_houses_truth["Hogwarts House"].to_numpy()

df_predictions = df_houses["Hogwarts House"].to_numpy()

match_array = (df_houses_truth == df_predictions)
match_count = np.sum(match_array)
total_count = match_array.size
accuracy = match_count / total_count
print(f"Accuracy of the predictions: {accuracy:.2%} ({match_count}/{total_count})")