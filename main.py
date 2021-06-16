from math import nan
from sklearn import utils
from models.LstmMVInput import LstmMVInput
import pandas as pd
from data.training_data import training_data, training_data_no_missing_values,training_data_extended_features_list,training_data_with_power

from models.KnnModel import KnnModel
from models.Linear import Linear
from models.KnnModel import KnnModel
from data.model_data import target_model_data
from data.isp1_results import get_power_generation,get_hydro_data
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
							res_total_json = get_json_for_line_fig(df,'Date','Res_Total'),
							load_total_json = get_json_for_line_fig(df,'Date','Load Total'),
							hydro_total_json = get_json_for_line_fig(df,'Date','Hydro Total'),
							sum_imports_json = get_json_for_line_fig(df,'Date','sum_imports'),
							sum_exports_json = get_json_for_line_fig(df,'Date','sum_exports'),
							)

@app.route('/Correlation')
def corrolations():

	df = get_data('training_data','*')
	
	hydro = pd.read_csv('mandatory_hydro.csv').set_index('Date').join((pd.read_csv('SMP_VALUES.csv').set_index('Date')))
	return render_template('correlation.jinja',title = 'Correlation',
							res_total_json = get_json_for_fig_scatter(df,'Res_Total','SMP'),
							load_total_json = get_json_for_fig_scatter(df,'Load Total','SMP'),
							hydro_total_json = get_json_for_fig_scatter(df,'Hydro Total','SMP'),
							sum_imports_json = get_json_for_fig_scatter(df,'sum_imports','SMP'),
							sum_exports_json = get_json_for_fig_scatter(df,'sum_exports','SMP'),
							man_hydro_json = get_json_for_fig_scatter(hydro,'Mandatory Hydro Injections','SMP')
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
	
	df['Prediction'], train_score, validation_score = KnnR.run_knn()
	
	return render_template('model.jinja', title = 'XgBoost Regresion Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/Lstm')
def lstm():
	df = get_data('training_data','*')
	# df = pd.read_csv('training_data_no_missing_values.csv')
	# df = training_data_with_power()
	# df = df.set_index('Date').join(pd.read_csv('mandatory_hydro.csv').set_index('Date')).join(pd.read_csv('power_generation.csv').set_index('Date'))
	lstm_model = LstmMVInput(utils.MAE,df)
	y_validation_prediction,hist_train,hist_val,lstm = lstm_model.run_lstm()
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
