import math
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import sqlalchemy


import plotly
import plotly.express as px
import json


import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

MSE = 'MSE'
MAE = 'MAE'
HuberLoss = 'HuberLoss'

extended_features_list = ['Date', 'Res_Total', 'Load Total', 'Hydro Total', 'sum_imports', 'sum_exports',
				 'weekdays', 'weekdays0', 'bankdays', 'bankdays0', 'winter', 'spring', 'autumn',
				 'summer', 't1_weekdays', ' t1_weekdays0', 't1_bankdays', 't1_bankdays0',
				 't1_winter', 't1_spring', 't1_autumn', 't1_summer', 'SMP']

features_list = ['Res_Total','Load Total','Hydro Total','Date','sum_imports','sum_exports','SMP']

def error_calculation(function, y_train, y_train_prediction, y_validation, y_validation_prediction):
	if MAE == function:
		# calculate mean absolute error
		train_score = mean_absolute_error(y_train_prediction.flatten(), y_train.flatten())
		print('Train Score: %.2f MAE' % train_score)
		test_score = mean_absolute_error(y_validation_prediction.flatten(), y_validation.flatten())
		print('Validation Score: %.2f MAE' % test_score)
		return[train_score,test_score]

	elif MSE == function:
		# calculate root mean squared error
		train_score = math.sqrt(mean_squared_error(y_train_prediction.flatten(), y_train.flatten()))
		print('Train Score: %.2f RMSE' % train_score)
		test_score = math.sqrt(mean_squared_error(y_validation_prediction.flatten(), y_validation.flatten()))
		print('Validation Score: %.2f RMSE' % test_score)
	return [train_score, test_score]


def loss_function_selection(function):
	if MAE == function:
		return torch.nn.L1Loss()
	elif MSE == function:
		return torch.nn.MSELoss(reduction='mean')
	elif HuberLoss == function:
		return torch.nn.SmoothL1Loss(reduction='mean')


def get_conn():
	engine = sqlalchemy.create_engine('sqlite:///database.db')
	return engine.connect()


def save_df_to_db(dataframe, df_name):
	connection = get_conn()
	dataframe.to_sql(df_name, connection, if_exists='replace')


def get_data(table, columns):
	connection = get_conn()
	return pd.read_sql(f'SELECT {columns} FROM {table}', connection)

def get_json_for_line_fig(df,x,y):
	fig = px.line(df,x=x,y=y)
	fig = fig.update_xaxes(rangeslider_visible=True)
	fig.update_layout(width=1500, height=500)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 

def get_json_for_fig_scatter(df,x,y):
	fig = px.scatter(df,x=x,y=y,trendline="ols")
	fig.update_layout(width=1500, height=500)
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 



def plot_lstm(data,lookback,y_train_prediction,y_validation_prediction,hist_train,hist_val):
	train_predict_plot = np.empty_like(data.iloc[:, -1])
	train_predict_plot[:] = np.nan
	train_predict_plot[lookback:len(y_train_prediction.flatten()) + lookback] = y_train_prediction.flatten()

	validation_predict_plot = np.empty_like(data.iloc[:, -1])
	validation_predict_plot[:] = np.nan
	validation_predict_plot[len(y_train_prediction.flatten()) + lookback : (int(len(data)/24)*24) ] = y_validation_prediction.flatten()

	fig, axs = plt.subplots(2)

	axs[0].plot(train_predict_plot, color='r', label='Train Prediction')

	axs[0].plot(validation_predict_plot, color='b', label='Validation Prediction')
	axs[0].plot(data.iloc[:, -1], color='y', label='Actual Price')
	axs[0].set_title('Model')
	axs[0].set_xlim(len(y_train_prediction.flatten()) - 25, len(y_train_prediction.flatten()) + 50)
	# axs[0].set_xlim(len(data)-50, len(data)+100)
	axs[0].legend()
	axs[1].plot(hist_train, label='Training Loss')
	axs[1].plot(hist_val, label='Validation Loss')
	axs[1].set_title('Loss')
	axs[1].legend()
	plt.show()
