import pandas as pd
import numpy as np
from describe import describe
import sys
from load_data import load_data

if len(sys.argv) < 2:
    print("Usage: python predict.py dataset.csv")
    sys.exit(1)
try:
    data = load_data(sys.argv[1])
except FileNotFoundError:
    print("File not found")
    sys.exit(1)
print(data.head())
description = describe(data)
#Drop not needed data
data = data.drop(columns=0)

#Convert diagnosis to binary
data["1_M"] = (data["1_M"] == True).astype(int)
data["1_B"] = (data["1_B"] == True).astype(int)

print ("Removing los outliers...")
z_threshold = 5
print(data.shape)
outliers = ((data - description.loc["mean",:]) / description.loc["std_dev",:]).abs() > z_threshold
cleaned_data = data[~outliers.any(axis=1)]
print(f"Removed {outliers.sum().sum()} outliers")

input("Press Enter to continue...")

cleaned_description = describe(cleaned_data)
for key in description.keys()[1:-2]:
    print(f"comparing outliers on column {key}")
    print((cleaned_data[cleaned_data[key] > cleaned_description[key]["upper 99.9% conf"]][key]), end = "\t")
    print((data[data[key] > description[key]["upper 99.9% conf"]][key]))
    print((cleaned_data[cleaned_data[key] < cleaned_description[key]["lower 99.9% conf"]][key]), end = "\t")
    print((data[data[key] < description[key]["lower 99.9% conf"]][key]))

for key in description.keys()[1:]:
    print(key, pd.concat([description[key], cleaned_description[key]], axis = 1))

cleaned_data.to_csv("data_cleaned.csv", index = False, index_label=0, columns = cleaned_data.columns)
print(cleaned_data.shape)