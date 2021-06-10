import datetime
from os import path

import pandas as pd

from utils.utils import extended_features_list, save_df_to_db


def date_con(date): return datetime.datetime.fromisoformat(date)


def training_data_no_missing_values():
	if path.exists('training_data_no_missing_values.csv'):
		return pd.read_csv('training_data_no_missing_values.csv')
	else:
		dataframe = pd.DataFrame(columns=extended_features_list)
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
		dataframe = dataframe.interpolate()
		dataframe = dataframe[extended_features_list]
		# del dataframe['Date']
		# del dataframe['Unnamed: 0']
		dataframe.to_csv('training_data_no_missing_values.csv', index=False)
		save_df_to_db(dataframe=dataframe, df_name='training_data_no_missing_values')


		return dataframe


def training_data():
	dataframe = pd.concat([pd.read_csv('data.csv',index_col=0), (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
	del dataframe['Unnamed: 0']
	dataframe.to_csv('data_for_training.csv', index=False)
	save_df_to_db(dataframe=dataframe, df_name='training_data')
	return dataframe


def training_data_extended_features_list():
	dataframe = pd.concat([pd.read_csv('FEATURES_USED.csv',index_col=0), (pd.read_csv('SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
	del dataframe['Unnamed: 0']
	dataframe.to_csv('training_data_extended_features_list.csv', index=False)
	save_df_to_db(dataframe=dataframe, df_name='training_data_extended_features_list')
	return dataframe

def training_data_with_power():
	data = pd.read_csv('data.csv',index_col=0)
	smp = pd.read_csv('SMP_VALUES.csv',index_col=0)
	power = pd.read_csv('power_plants.csv',index_col=0)
	return data.set_index('Date').join(smp.set_index('Date')).join(power.set_index('Date'))#Fix this Daylight saving