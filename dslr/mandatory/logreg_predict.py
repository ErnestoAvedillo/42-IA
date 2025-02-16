import pandas as pd
import sys
import numpy as np
import json
from logistic_prediction import predict

if len(sys.argv) != 2:
    print("Please give a file as argument to describe.")
    exit(1)
try:
    df = pd.read_csv(sys.argv[1])
except:
    print("The file you entered does not exist or you don't have access.")
    exit(1)
#get the arguments and the options for the houses
arguments_file = "arguments.json"
with open(arguments_file, "r", encoding="utf-8") as my_file:
    data = json.load(my_file)
    theta = np.array(data["arguments"])
with open("houses.json", "r", encoding="utf-8") as myfile:
    data = json.load(myfile)
    options = data["houses"]
# remove all the rows with missing values
df = df.dropna(axis=1, how = 'all')
df = df.dropna(axis = 0)

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
list_of_houses = predict(X, theta,options)

#save the predictions in a csv file
df_houses = pd.DataFrame(list_of_houses, columns = ["Hogwarts House"])
df_houses.to_csv("houses.csv", index = True, index_label="Index")