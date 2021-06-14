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
	

	df['Prediction'], train_score, validation_score = linear.run_linear()

	return render_template('model.jinja', title = 'Linear Model Last 24hours Pediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/Knn')
def Knn():
	df = get_data('training_data','*')
	KnnR = KnnModel(data=df)
	
	df['Prediction'], train_score, validation_score = KnnR.run_knn()
	
	return render_template('model.jinja', title = 'KnnR Model Last 24hours Pediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)


@app.route('/XgB')
def XgB():
	df = get_data('training_data','*')
	KnnR = KnnModel(data=df)
	
	df['Prediction'], train_score, validation_score = KnnR.run_knn()
	
	return render_template('model.jinja', title = 'XgBoost Regresion Last 24hours Pediction vs Actual Price',
							chart_json = get_json_for_line_fig(df[-24:],'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/Lstm')
def lstm():
	df = get_data('training_data','*')
	# df = pd.read_csv('training_data_no_missing_values.csv')
	# df = training_data_with_power()
	df = df.set_index('Date').join(pd.read_csv('mandatory_hydro.csv').set_index('Date'))
	lstm_model = LstmMVInput(utils.MAE,df)
	lstm_model.run_lstm()
	return render_template('base.jinja')


if __name__ == '__main__':

	app.run(host="localhost", port=8000, debug=True)
