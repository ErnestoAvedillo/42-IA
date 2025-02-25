import pandas as pd
import numpy as np
from describe import describe
import sys

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0, header=None)
    data = data.dropna()
    data = data.drop_duplicates()
    data = pd.get_dummies(data)
    return data


if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1])
except FileNotFoundError:
    print("File not found")
    sys.exit(1)
print("imprimo los datos")
print(data.head())
input("Press enter...")
print("imprimo la descripcion de los datos")
print("imprimo las keys  de los datos")
keys = data.keys()
print (keys)
#show_all_histograms(data)
for key in keys:
    if data[key].value_counts().count() > 10:
        continue
    print(key, data[key].unique())
    print(key, data[key].value_counts())

description = describe(data)
z_threshold = 4
data = data.drop(columns=["1_M"])
print(data.shape)
data =data
outliers = ((data - description.loc["mean",:]) / description.loc["std_dev",:]).abs() > z_threshold
cleaned_data = data[~outliers.any(axis=1)]
print(f"Removed {outliers.sum().sum()} outliers")

description = describe(cleaned_data)
for key in description.keys()[1:-2]:
    print((cleaned_data[cleaned_data[key] > description[key]["upper 99.9% conf"]]))
    print((cleaned_data[cleaned_data[key] < description[key]["lower 99.9% conf"]]))

for key in description.keys()[1:]:
    print(key, description[key])

cleaned_data.to_csv("data_cleaned.csv", index = False, index_label=0, columns = cleaned_data.columns)
print(cleaned_data.shape)