import flask
from flask import render_template

from utils.database_interface import DB
from utils.web_utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics,get_json_for_line_scatter
from datetime import date
import pandas as pd
import json

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
@app.route('/')
def index():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('Date')
	return render_template('charts.jinja',title = 'Train Data For The Past 7 Days',df=df,get_json = get_json_for_line_fig,x=df.index)

@app.route('/Correlation')
def corrolations():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('SMP')
	df = df.iloc[:,df.columns!='Date'].dropna()
	return render_template('charts.jinja',title = 'Correlation',df=df,get_json = get_json_for_fig_scatter,x=df.index)

@app.route('/Linear')
def Linear_page():
	db = DB()
	df = db.get_data('*','linear').set_index('Date')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('linear',db).iloc[0]

	return render_template('model.jinja', title = 'Linear Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error']
							)

@app.route('/KnnR')
def Knn():
	db = DB()
	df = db.get_data('*','Knn').set_index('Date')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('Knn',db).iloc[0]
	return render_template('model.jinja', title = 'KnnR Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])


@app.route('/XgB')
def XgB():
	db = DB()
	df = db.get_data('*','XgB').set_index('Date')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	metrics = get_metrics('XgB',db).iloc[0]
	return render_template('model.jinja', title = 'XgB Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])

@app.route('/Lstm')
def lstm():
	db = DB()
	df = db.get_data('*','Lstm').set_index('Date')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	hist = db.get_data('*','hist_lstm')

	metrics = get_metrics('Lstm',db).iloc[0]
	return render_template('lstm.jinja', title = 'Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'],
							hist_json = get_json_for_line_scatter(hist,hist.index,['hist_train','hist_val']))

@app.route('/Hybrid_Lstm')
def hybrid_lstm():
	db = DB()
	df = db.get_data('*','Hybrid_Lstm')

	if (df.columns == 'Inference').any():
		y=['Prediction','SMP','Inference']
	else:
		y=['Prediction','SMP']

	hist = db.get_data('*','hist_Hybrid_Lstm')

	metrics = get_metrics('Hybrid_Lstm',db).iloc[0]
	return render_template('lstm.jinja', title = 'Hybrid Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,'Date',y),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'],
							hist_json = get_json_for_line_scatter(hist,hist.index,['hist_train','hist_val']))

@app.route('/api')
def api():
	try:
		db = DB()
		df = {}
		df['Date'] = db.get_data('*','linear')[-24:]['Date'].astype(str).to_list()
		df['Linear'] = db.get_data('*','linear').set_index('Date')['Inference'].dropna().to_list()
		df['Knn'] = db.get_data('*','Knn').set_index('Date')['Inference'].dropna().to_list()
		df['XgB'] = db.get_data('*','XgB').set_index('Date')['Inference'].dropna().to_list()
		df['Lstm'] = db.get_data('*','Lstm').set_index('Date')['Inference'].dropna().to_list()
		df['Hybrid_Lstm'] = db.get_data('*','Hybrid_Lstm').set_index('Date')['Inference'].dropna().to_list()
		return json.dumps(df)
	except:
		return 'No Prediction Possible'



if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
