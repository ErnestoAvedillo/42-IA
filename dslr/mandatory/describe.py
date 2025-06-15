import argparse
import numpy as np
import pandas as pd
import math
import sys

def get_percentils(percent, sorted_column, count):
	
	raw_position = (count - 1) * percent
	lowest_position = math.floor(raw_position)
	highest_position = math.ceil(raw_position)
	
	lower_value = sorted_column[lowest_position]
	upper_value = sorted_column[highest_position]
	decimal_part = raw_position - lowest_position
	
	return lower_value + (decimal_part * (upper_value - lower_value))

def my_mean(data:np.ndarray):
	sum = 0
	max =-math.inf
	min = math.inf
	count = 0
	for value in data:
		if type(value) == str:
			value = len(value)
		else:
			if np.isnan(value):
				continue
		if value > max: 
			max = value
		if value < min:
			min = value
		sum += value
		count += 1
	if max == -math.inf:
		max = None
	if min == math.inf:
		min = None
	if count == 0:
		return np.nan, count, np.nan, np.nan, np.nan
	return sum/count, count, sum, max, min
def my_std_dev(data:np.ndarray):
	mean, count, _, _, _ = my_mean(data)
	cuadratic_sum = 0
	for value in data:
		if np.isnan(value):
			continue
#		if type(value) == str:
#		cuadratic_sum += math.pow(len(value) - mean, 2)
#	else:
		cuadratic_sum += math.pow(value - mean, 2)
		if np.isnan(cuadratic_sum):
			print(f"cuadratic_sum is nan: {value}-- mean")		
			input("Press Enter to continue...")
	return math.sqrt(cuadratic_sum / count)

def describe (data:pd.DataFrame):
	keys = data.keys()
	num_col = len(keys)
	stats_data = {}
	for col in range(num_col):
		if data[keys[col]].dtypes == 'object' or data[keys[col]].dtypes == 'str':
			continue
		data_col = data[keys[col]].dropna().to_numpy()
		mean, count, sum, max, min = my_mean(data_col)
		try:
			std_dev = my_std_dev(data_col)
			amplitude = max - min
			
			val25 = get_percentils(0.25, sorted(data_col), count)
			val50 = get_percentils(0.5, sorted(data_col), count)
			val75 = get_percentils(0.75, sorted(data_col), count)

		except:
			std_dev = np.nan
			amplitude = np.nan
			val25 = np.nan
			val50 = np.nan
			val75 = np.nan
		stats_data[keys[col]] =[count, mean, std_dev, min, val25, val50, val75, max]
	stats_df = pd.DataFrame(stats_data, index=["count", "mean", "std_dev", "min", "25%", "50%", "75%", "max"])
	return stats_df

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
	data=df
	print(df.describe())
	resutl = describe(data)
	print (resutl)