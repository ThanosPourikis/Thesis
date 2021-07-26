import datetime
import math
# from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.get_weather_data import get_weather_data

MSE = 'MSE'
MAE = 'MAE'
HuberLoss = 'HuberLoss'

#features_list = ['Res_Total','Load Total','Hydro Total','Date','sum_imports','sum_exports','SMP']

def error_calculation(function, y_train, y_train_prediction, y_validation, y_validation_prediction):
	if MAE == function:
		# calculate mean absolute error
		train_score = mean_absolute_error(y_train_prediction.flatten(), y_train.flatten())
		print('Train Score: %.2f MAE' % train_score)
		test_score = mean_absolute_error(y_validation_prediction.flatten(), y_validation.flatten())
		print('Validation Score: %.2f MAE' % test_score)
		return train_score,test_score

	elif MSE == function:
		# calculate root mean squared error
		train_score = math.sqrt(mean_squared_error(y_train_prediction.flatten(), y_train.flatten()))
		print('Train Score: %.2f RMSE' % train_score)
		test_score = math.sqrt(mean_squared_error(y_validation_prediction.flatten(), y_validation.flatten()))
		print('Validation Score: %.2f RMSE' % test_score)
	return train_score, test_score

def get_req():
	return pd.read_csv('datasets/requirements.csv').set_index('Date').join(pd.read_csv('datasets/SMP.csv').set_index('Date')).reset_index()

def get_data_from_csv():
	return pd.read_csv('datasets/requirements.csv').set_index('Date').join(pd.read_csv('datasets/units.csv').set_index('Date')).join(pd.read_csv('datasets/SMP.csv').set_index('Date')).reset_index()
	
def get_data_from_csv_with_weather():
	return (get_data_from_csv().set_index('Date').join(get_weather_data().set_index('Date'))).reset_index()

def date_con(date): return datetime.datetime.fromisoformat(date)

def check_data_integrity(df):
	print(len(df)/24)
	df.to_csv('check_set')
	df = df['Date']
	delt = datetime.timedelta(hours= 1)
	for i in range(len(df)-1):
		if date_con(df[i]) + delt != date_con(df[i+1]):
			print (f'Error at {df[i]}')

def save_metrics(metrics,model,db):
	try:
		metrics = pd.concat([metrics,db.get_data('*',f'metrics_{model}')])
		metrics = metrics.reset_index().drop_duplicates(subset ='index',keep='last').set_index('index')
		db.save_df_to_db(metrics,f'metrics_{model}')
	except:
		metrics = metrics.reset_index().drop_duplicates(subset ='index',keep='last').set_index('index')
		db.save_df_to_db(metrics,f'metrics_{model}')


