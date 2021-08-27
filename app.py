import flask
from flask import render_template

from utils.database_interface import DB
from utils.web_utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics,get_json_for_line_scatter,get_candlesticks
from datetime import date
import pandas as pd
import json

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
models = ['Linear','KnnR','XgB','Lstm','Hybrid_Lstm']

def page(name):
	db = DB()
	df = db.get_data('*',name)
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce')[name]

	metrics = get_metrics(name,db)

	return render_template('model.jinja', title = f'{name} Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics
							)

@app.route('/')
def index():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('Date')
	return render_template('home.jinja',title = 'Train Data For The Past 7 Days',df=df,get_json = get_json_for_line_fig,candlestick = get_candlesticks(df.SMP))

@app.route('/Correlation')
def corrolations():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('SMP')
	df = df.iloc[:,df.columns!='Date'].dropna()
	return render_template('correlation.jinja',title = 'Correlation For The Past 7 Days',df=df,get_json = get_json_for_fig_scatter)

@app.route('/Linear')
def linear_page():
	return page('Linear')

@app.route('/Knn')
def knn():
	return page('Knn')



@app.route('/XgB')
def xgB():
	return page('XgB')

@app.route('/Lstm')
def lstm():
	db = DB()
	df = db.get_data('*','Lstm')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce')['Lstm']

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
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce')['Hybrid_Lstm']
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
@app.route('/metrics_api/<algo>')
def metrics_api(algo):
	
	db = DB()
	try : 
		return get_metrics(algo,db).to_json()
	except:
		return "WRONG"
if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
