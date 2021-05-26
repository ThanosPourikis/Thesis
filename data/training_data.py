import datetime

import pandas as pd


def date_con(date): return datetime.datetime.fromisoformat(date)


def training_data_no_missing_values():

    dataframe = pd.DataFrame(columns=['Res_Total', 'Load Total', 'Hydro Total', 'Date', 'sum_imports', 'sum_exports'])
    data = pd.concat([pd.read_csv('data.csv'), (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
    temp = {'Res_Total': 0,
            'Load Total': 0,
            'Hydro Total': 0,
            'Date': 0,
            'sum_imports': 0,
            'sum_exports': 0,
            'SMP': 0}
    start_date = date_con('2020-11-13 00:00:00+01:00')
    i = 0
    del data['Unnamed: 0']
    while start_date != date_con(data.iloc[-1].loc['Date']):
        if start_date == date_con(data.loc[i, 'Date']):
            dataframe = dataframe.append(data.loc[i])
            i += 1
        else:
            temp['Date'] = start_date
            dataframe = dataframe.append(temp, ignore_index=True)
            print(f'Adding missing date {start_date}')
        start_date += datetime.timedelta(hours=1)

    del dataframe['Date']
    # del dataframe['Unnamed: 0']
    dataframe.to_csv('data_for_training.csv', index=False)

    return dataframe


def training_data():
    return pd.concat([pd.read_csv('data.csv'), (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
