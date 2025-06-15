import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Describe the CSV to be described with custom statistics.")
	parser.add_argument("-f","--file", type=str, help="Path to the CSV file to describe.")
	args = parser.parse_args()

	if not args.file:
		print("Please give a file as argument to describe.")
		exit(1)
	try:
		df = pd.read_csv(args.file)
	except:
		print("The file you entered does not exist or you don't have access.")
		exit(1)

	data = df.dropna(axis=1, how = 'all')
	
	print (data.describe())
	sns.pairplot(data)
	plt.show()
