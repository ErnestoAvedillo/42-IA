import numpy as np
import pandas as pd
import math
import sys
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
		cuadratic_sum += math.pow(value - mean, 2)
		if np.isnan(cuadratic_sum):
			print(f"cuadratic_sum is nan: {value}-- mean")		
			input("Press Enter to continue...")
	return math.sqrt(cuadratic_sum / count)

def describe (data:pd.DataFrame):
	if data.ndim == 1:
		data = data.to_frame()
	keys = data.keys()
	num_col = len(keys)
	stats_data = {}
	z_threshold = 4
	for col in range(num_col):
		if data[keys[col]].dtypes == 'object' or data[keys[col]].dtypes == 'str':
			continue
		mean, count, sum, max, min = my_mean(data[keys[col]].to_numpy())
		try:
			std_dev = my_std_dev(data[keys[col]].to_numpy())
			amplitude = max - min
			val25 = min + 0.25 * amplitude
			val50 = min + 0.50 * amplitude
			val75 = min + 0.75 * amplitude
		except:
			std_dev = np.nan
			amplitude = np.nan
			val25 = np.nan
			val50 = np.nan
			val75 = np.nan
		stats_data[keys[col]] =[
			count, 
			mean, 
			std_dev, 
			min, 
			val25, 
			val50, 
			val75, 
			max, 
			mean - 1.96 * std_dev, 
			mean + 1.96 * std_dev, 
			mean - 3.291 * std_dev, 
			mean + 3.291 * std_dev,
			((data[keys[col]] < mean - z_threshold * std_dev) | (data[keys[col]] > mean + z_threshold * std_dev)).sum()]
	stats_df = pd.DataFrame(stats_data, index=["count", "mean", "std_dev", "min", "25%", "50%", "75%", "max", "lower 95% conf", "upper 95% conf", "lower 99.9% conf", "upper 99.9% conf", "Outliers (5s)"])
	return stats_df
