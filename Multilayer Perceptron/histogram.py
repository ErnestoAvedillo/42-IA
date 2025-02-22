import numpy as  np
from matplotlib import pyplot as plt
import pandas as pd
import sys

def show_all_histograms(data:pd.DataFrame):
	keys = data.keys()
	courses = data["Hogwarts House"].unique()
	color_list = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
	n_graficas = len(keys)
	filas = int(np.ceil(n_graficas ** 0.5)) 
	columnas = int(np.ceil(n_graficas / filas)) 
	fig, axes = plt.subplots(filas, columnas, figsize=(12, 8))
	axes = axes.flatten()
	i = 0
	for key in keys:
		j = 0
		for course in courses:
			subset= data[data["Hogwarts House"] == course][key]
			axes[i].hist(subset.to_numpy(), bins = 10, color=color_list[j], edgecolor= "black", alpha = 0.7)
			j += 1
		axes[i].set_title(key)
		i +=1
	for i in range(n_graficas, len(axes)):
		fig.delaxes(axes[i])
	plt.tight_layout()
	plt.show()

if len(sys.argv) != 2:
	print("Please give a file as argument to describe.")
	exit(1)
try:
	df = pd.read_csv(sys.argv[1])
except:
	print("The file you entered does not exist or you don't have access.")
	exit(1)

data = df.dropna(axis=1, how = 'all')
show_all_histograms(data)
