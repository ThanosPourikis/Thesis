import flask
from flask import render_template

from utils.database_interface import DB
from utils.utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics
from datetime import date
import pandas as pd


app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
@app.route('/')
def index():
	db = DB()
	df = db.get_data('*', 'dataset')
	return render_template('charts.jinja',title = 'Train Data For The Past 7 Days',df=df[-(7*24):],get_json = get_json_for_line_fig,x='Date')

@app.route('/Correlation')
def corrolations():
	db = DB()
	df = db.get_data('*', 'dataset')
	df = df.iloc[:,df.columns!='Date'].dropna()
	return render_template('charts.jinja',title = 'Correlation',df=df[-(7*24):],get_json = get_json_for_fig_scatter,x='SMP')

@app.route('/Linear')
def Linear_page():
	db = DB()
	df = db.get_data('*','linear')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('linear',db).iloc[0]

	return render_template('model.jinja', title = 'Linear Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_fig(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error']
							)

@app.route('/KnnR')
def Knn():
	db = DB()
	df = db.get_data('*','Knn')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('Knn',db).iloc[0]
	return render_template('model.jinja', title = 'KnnR Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_fig(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])


@app.route('/XgB')
def XgB():
	db = DB()
	df = db.get_data('*','XgB')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('XgB',db).iloc[0]
	return render_template('model.jinja', title = 'XgB Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_fig(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])

@app.route('/Lstm')
def lstm():
	db = DB()
	df = db.get_data('*','Lstm')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	hist = db.get_data('*','hist_lstm')

	metrics = get_metrics('Lstm',db).iloc[0]
	return render_template('lstm.jinja', title = 'Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_fig(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'],
							hist_json = get_json_for_line_fig(hist,hist.index,['hist_train','hist_val']))

if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
