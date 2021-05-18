
import pandas as pd

from data.model_data import target_model_data


def training_data():

    dataframe = target_model_data()
    temp = pd.read_csv('SMP_VALUES.csv')
    dataframe["SMP"] = temp['SMP']
    del dataframe['Date']
    dataframe.to_csv('data.csv', index=False)

    return dataframe
