from math import nan
from sklearn import utils
from models.LstmMVInput import LstmMVInput
import pandas as pd
from data.training_data import update_data

from models.KnnModel import KnnModel
from models.Linear import Linear
from models.KnnModel import KnnModel
from data.model_data import target_model_data
from data.isp1_results import get_excel_data
import flask
import sqlalchemy
from flask import render_template

import utils.utils as utils
from utils.utils import get_data, get_json_for_line_fig,get_json_for_fig_scatter

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config['Debug'] = True


@app.route('/')
def index():
	df = get_data('training_data','*')
	return render_template('home.jinja',title = 'Original Data',
							smp_json = get_json_for_line_fig(df,'Date','SMP'),
							res_total_json = get_json_for_line_fig(df,'Date','Renewables'),
							load_total_json = get_json_for_line_fig(df,'Date','System Load'),
							hydro_total_json = get_json_for_line_fig(df,'Date','Mandatory_Hydro'),
							sum_imports_json = get_json_for_line_fig(df,'Date','imports'),
							sum_exports_json = get_json_for_line_fig(df,'Date','export'),
							)

@app.route('/Correlation')
def corrolations():

	df = get_data('training_data','*')
	return render_template('correlation.jinja',title = 'Correlation',
							res_total_json = get_json_for_fig_scatter(df,'Renewables','SMP'),
							load_total_json = get_json_for_fig_scatter(df,'System Load','SMP'),
							hydro_total_json = get_json_for_fig_scatter(df,'Mandatory_Hydro','SMP'),
							sum_imports_json = get_json_for_fig_scatter(df,'imports','SMP'),
							sum_exports_json = get_json_for_fig_scatter(df,'export','SMP'),
							)

@app.route('/Linear')
def Linear_page():
	df = get_data('training_data','*')

	linear = Linear(data=df)
	

	prediction, train_score, validation_score = linear.run()
	df['Prediction'] = nan
	df[-len(prediction):]['Prediction'] = prediction
	df = df.dropna()

	return render_template('model.jinja', title = 'Linear Model Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/KnnR')
def Knn():
	df = get_data('training_data','*')
	KnnR = KnnModel(data=df)
	
	prediction, train_score, validation_score = KnnR.run()
	df['Prediction'] = nan
	df[-len(prediction):]['Prediction'] = prediction
	df = df.dropna()

	return render_template('model.jinja', title = 'KnnR Model Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)


@app.route('/XgB')
def XgB():
	df = get_data('training_data','*')
	KnnR = KnnModel(data=df)
	
	df['Prediction'], train_score, validation_score = KnnR.run()
	
	return render_template('model.jinja', title = 'XgBoost Regresion Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/Lstm')
def lstm():
	df = get_data('training_data','*')
	# df = pd.read_csv('datasets/training_data_no_missing_values.csv')
	# df = training_data_with_power()
	# df = (pd.read_csv('datasets/SMP.csv').set_index('Date')).join(pd.read_csv('datasets/power_generation.csv').set_index('Date'))
	lstm_model = LstmMVInput(utils.MAE,df)
	y_validation_prediction,hist_train,hist_val,lstm = lstm_model.run()
	df['Prediction'] = nan
	df[-len(y_validation_prediction):]['Prediction'] = y_validation_prediction
	df = df.dropna()
	hist = pd.DataFrame({'hist_train': hist_train ,'hist_val':hist_val})

	return render_template('lstm.jinja',title = 'Lstm Last 24hour Prediction vs Actual',
												train_score = lstm[0],
												validation_score = lstm[1],
												train_time = lstm[2],
												chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
												hist_json = get_json_for_line_fig(hist,hist.index,['hist_train','hist_val'])
												)


if __name__ == '__main__':

	app.run(host="localhost", port=8000, debug=True)
