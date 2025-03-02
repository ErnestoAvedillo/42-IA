import pandas as pd


def load_data(file_path, index_col=None, header=None):
    data = pd.read_csv(file_path, index_col=index_col, header=header)
    data = data.dropna()
    data = data.drop_duplicates()
    data = pd.get_dummies(data)
    return data
