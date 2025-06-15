from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse


def scatter (data:pd.DataFrame):
	keys = data.keys()
	n_graficas = np.arange(2,len(keys)-4).sum()
	groups = data["Hogwarts House"].unique()
	color_list = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
	filas = int(np.ceil(n_graficas ** 0.5)) 
	columnas = int(np.ceil(n_graficas / filas)) 
	fig, axes = plt.subplots(filas, columnas, figsize=(24, 16))
	axes=axes.flatten()
	k = 0
	for i in range(5,len(keys)):
		for j in range (i+1,len(keys)):
			n_color = 0
			for group in groups:
				subset= data[data["Hogwarts House"] == group]
				#plt.scatter(subset[keys[i]].to_numpy(),subset[keys[j]].to_numpy(),color=color_list[n_color])
				axes[k].scatter(subset[keys[i]].to_numpy(),subset[keys[j]].to_numpy(),color=color_list[n_color])
				n_color += 1
			#plt.title(f"{keys[i]} vs. {keys[j]}")
			#plt.show()
			axes[k].set_xticks([])
			axes[k].set_yticks([])
			axes[k].set_title(f"{keys[i]} vs. {keys[j]}", fontsize=7)
			k += 1
	for i in range(k, len(axes)):
		fig.delaxes(axes[i])
	plt.tight_layout()
	plt.subplots_adjust()
	plt.show()

if __name__ == "__main__":
	parserr = argparse.ArgumentParser(description="The CSV file to be graphed with custom statistics.")
	parserr.add_argument("-f","--file", type=str, help="Path to the CSV file to graph.")
	args = parserr.parse_args()
if not args.file:
	print("Please give a file as argument to graph.")
	exit(1)
try:
	df = pd.read_csv(args.file)
except:
	print("The file you entered does not exist or you don't have access.")
	exit(1)
resutl = scatter(df)
