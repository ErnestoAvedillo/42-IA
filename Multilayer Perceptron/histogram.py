import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", etc., depending on your system
import numpy as  np
from matplotlib import pyplot as plt
import pandas as pd
import sys
from load_data import load_data
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from describe import describe

def show_all_histograms(data:pd.DataFrame):
	keys = data.keys()
	outputs = data[keys[31]].unique()
	color_list = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
	root = tk.Tk()
	root.title("Histograms with describe info")
	tab_control = ttk.Notebook(root)
	for key in keys:
		tab = ttk.Frame(tab_control)
		tab_control.add(tab, text = key)
		j = 0
		fig, axe = plt.subplots(figsize=(6,5))
		for output in outputs:
			subset= data[data[keys[31]] == output][key]
			subset.hist(bins = 10, color=color_list[j], edgecolor= "black", alpha = 0.7)
			j += 1
		axe.set_title(f"Histogram for key: {key}")
		axe.set_xlabel(key)
		axe.set_ylabel("Frequency")
		#Add text oputput
		aux = data[key]
		stats = describe(aux)
		textstr = '\n'.join([f'{index}: {value:.2f}' for index, value in stats[key].items()])
		axe.text(0.95, 0.95, textstr, transform=axe.transAxes, fontsize=10,
			           verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
		#Embed the figure in the tab
		canvas = FigureCanvasTkAgg(fig, master=tab)
		canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		canvas.draw()
	# Pack and run the GUI
	tab_control.pack(expand=1, fill="both")

	def on_closing():
		root.quit()
		root.destroy()

	root.protocol("WM_DELETE_WINDOW", on_closing)
	root.mainloop()
	print("Finished")


if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1])
except FileNotFoundError:
    print("File not found")
    sys.exit(1)
show_all_histograms(data)