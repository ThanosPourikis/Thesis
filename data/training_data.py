
from model_data import target_model_data
import pandas as pd


def training_data():

    dataframe = target_model_data()
    temp = pd.read_csv('../SMP_VALUES.csv')
    dataframe["SMP"] = temp['SMP']
    del dataframe['Date']
    dataframe.to_csv('data.csv', index=False)

    return dataframe
