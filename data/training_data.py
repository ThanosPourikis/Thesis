import datetime
from os import path

import pandas as pd

from utils.utils import features_list, save_to_db


def date_con(date): return datetime.datetime.fromisoformat(date)


def training_data_no_missing_values():
    if path.exists('data_for_training.csv'):
        return pd.read_csv('data_for_training.csv')
    else:
        dataframe = pd.DataFrame(columns=features_list)
        data = pd.concat([pd.read_csv('FEATURES_USED.csv', index_col=0),
                          (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
        start_date = date_con('2020-11-13 00:00:00+01:00')
        i = 0
        while start_date.ctime() != date_con(data.iloc[-1].loc['Date']).ctime():
            if start_date.ctime() == date_con(data.loc[i, 'Date']).ctime():
                dataframe = dataframe.append(data.iloc[i])
                i += 1
            else:
                print(f'Adding missing date {start_date}')
                dataframe = dataframe.append({'Date': start_date}, ignore_index=True)
            start_date += datetime.timedelta(hours=1)
        dataframe = dataframe.fillna(0)
        dataframe = dataframe[features_list]
        # del dataframe['Date']
        # del dataframe['Unnamed: 0']
        dataframe.to_csv('data_for_training.csv', index=False)

        return dataframe


def training_data():
    dataframe = pd.concat([pd.read_csv('data.csv'), (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
    del dataframe['Unnamed: 0']
    save_to_db(dataframe=dataframe, df_name='training_data')
    return dataframe
