from os import name
import flask
from flask import render_template

from utils.database_interface import DB
from utils.web_utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics,get_json_for_line_scatter,get_candlesticks
from datetime import date, timedelta
import pandas as pd
import json
from pytz import timezone


localTz = timezone('CET')

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
week = str(localTz.localize(today - timedelta(weeks=1)))
models = ['Linear','KnnR','XgB','Lstm','Hybrid_Lstm']
dataset = 'requirements'


@app.route('/')
def index():
	db = DB()
	df = db.get_data('*', dataset,f'"index" > "{week}"')
	return render_template('home.jinja',title = 'Train Data For The Past 7 Days',df=df,get_json = get_json_for_line_fig,candlestick = get_candlesticks(df.SMP))

@app.route('/Correlation')
def corrolations():
	db = DB()
	df = db.get_data('*', dataset,f'"index" > "{week}"').set_index('SMP').dropna()
	return render_template('correlation.jinja',title = 'Correlation For The Past 7 Days',df=df,get_json = get_json_for_fig_scatter)

@app.route('/<name>')
def page_for_ml_model(name):
	db = DB()
	df = db.get_data('*',name)
	df['Previous Prediction'] = db.get_data(f'"index","{name}"','infernce')

	metrics = get_metrics(name,db)

	return render_template('model.jinja', title = f'{name} Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics
							)

@app.route('/Lstm')
def lstm():
	db = DB()
	df = db.get_data('*','Lstm')
	df['Previous Prediction'] = db.get_data('"index","Lstm"','infernce')
	hist = db.get_data('*','hist_lstm')

	metrics = get_metrics('Lstm',db)
	return render_template('lstm.jinja', title = 'Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics.iloc[0]['best_epoch']))

@app.route('/Hybrid_Lstm')
def hybrid_lstm():
	db = DB()
	df = db.get_data('*','Hybrid_Lstm')
	df['Previous Prediction'] = db.get_data('"index","Hybrid_Lstm"','infernce')
	hist = db.get_data('*','hist_Hybrid_Lstm')
	
	metrics = get_metrics('Hybrid_Lstm',db)
	return render_template('lstm.jinja', title = 'Hybrid Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics.iloc[0]['best_epoch']))

@app.route('/prices_api')
def prices_api():
	try:
		db = DB()
		df = pd.DataFrame()
		df['Linear'] = db.get_data('"index","Inference"','linear').dropna()
		df['Knn'] = db.get_data('"index","Inference"','Knn').dropna()
		df['XgB'] = db.get_data('"index","Inference"','XgB').dropna()
		df['Lstm'] = db.get_data('"index","Inference"','Lstm').dropna()
		df['Hybrid_Lstm'] = db.get_data('"index","Inference"','Hybrid_Lstm').dropna()
		return df.reset_index(drop=True).to_json()
	except:
		return 'No Prediction Possible'
@app.route('/metrics_api/<algo>')
def metrics_api(algo):
	
	db = DB()
	try : 
		return get_metrics(algo,db).to_json()
	except:
		return "WRONG"
if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
