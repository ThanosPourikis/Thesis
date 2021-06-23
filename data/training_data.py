from data.isp1_results import get_excel_data
from data.get_SMP_data import get_SMP_data
import datetime
from os import path

import pandas as pd

from utils.utils import save_df_to_db


def date_con(date): return datetime.datetime.fromisoformat(date)


# def training_data_no_missing_values():
# 	if path.exists('training_data_no_missing_values.csv'):
# 		return pd.read_csv('datasets/training_data_no_missing_values.csv')
# 	else:
# 		dataframe = pd.DataFrame(columns=extended_features_list)
# 		data = pd.concat([pd.read_csv('datasets/FEATURES_USED.csv', index_col=0),
# 						  (pd.read_csv('datasets/SMP_VALUES.csv'))['SMP']], axis=1, join='inner')
# 		start_date = date_con('2020-11-13 00:00:00+01:00')
# 		i = 0
# 		while start_date.ctime() != date_con(data.iloc[-1].loc['Date']).ctime():
# 			if start_date.ctime() == date_con(data.loc[i, 'Date']).ctime():
# 				dataframe = dataframe.append(data.iloc[i])
# 				i += 1
# 			else:
# 				print(f'Adding missing date {start_date}')
# 				dataframe = dataframe.append({'Date': start_date}, ignore_index=True)
# 			start_date += datetime.timedelta(hours=1)
# 		dataframe = dataframe.interpolate()
# 		dataframe = dataframe[extended_features_list]
# 		# del dataframe['Date']
# 		# del dataframe['Unnamed: 0']
# 		dataframe.to_csv('datasets/training_data_no_missing_values.csv', index=False)
# 		save_df_to_db(dataframe=dataframe, df_name='training_data_no_missing_values')

# 		return dataframe


def training_data():
	data = pd.read_csv('datasets/power.csv',index_col=0)
	smp = pd.read_csv('datasets/SMP.csv',index_col=0)
	dataframe = data.join(smp)
	dataframe = dataframe.dropna()
	dataframe.to_csv('datasets/data_for_training.csv')
	save_df_to_db(dataframe=dataframe, df_name='training_data')
	dataframe['Date'] = dataframe.index
	return dataframe


def training_data_extended_features_list():
	data = pd.read_csv('datasets/FEATURES_USED.csv',index_col=0)
	smp = pd.read_csv('datasets/SMP.csv',index_col=0)
	dataframe = data.set_index('Date').join(smp)
	dataframe.to_csv('datasets/data_for_training_extended_features_list.csv')
	save_df_to_db(dataframe=dataframe, df_name='data_for_training_extended_features_list')
	dataframe['Date'] = dataframe.index
	return dataframe

def training_data_with_power():
	data = pd.read_csv('datasets/power.csv',index_col=0)
	power = pd.read_csv('datasets/power_generation.csv',index_col=0)
	smp = pd.read_csv('datasets/SMP.csv',index_col=0)
	dataframe =  data.join(power).join(smp)
	dataframe.to_csv('datasets/data_for_training_power_generation.csv',)
	save_df_to_db(dataframe=dataframe, df_name='data_for_training_power_generation')
	dataframe['Date'] = dataframe.index
	return dataframe

def update_data(new_files = False):
	print('Don\'t panic if it looks frozen')
	get_SMP_data(new_files)
	smp = pd.read_csv('datasets/SMP.csv').set_index('Date')
	get_excel_data()
	power = pd.read_csv('datasets/power.csv').set_index('Date')
	gen = pd.read_csv('datasets/power_generation.csv').set_index('Date')
	export = power.join(gen).join(smp).dropna()
	
	save_df_to_db(export,'training_data')
	