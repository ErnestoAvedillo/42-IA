import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import sys


if len(sys.argv) != 2:
	print("Please give a file as argument to describe.")
	exit(1)
try:
	df = pd.read_csv(sys.argv[1])
except:
	print("The file you entered does not exist or you don't have access.")
	exit(1)

data = df.dropna(axis=1, how = 'all')

print (data.describe())
sns.pairplot(data)
plt.show()
