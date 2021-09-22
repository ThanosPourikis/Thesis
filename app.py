import flask
from flask import render_template,Flask
from werkzeug.utils import redirect

from utils.database_interface import DB
from utils.web_utils import *
from datetime import date, timedelta
import pandas as pd
from pytz import timezone


localTz = timezone('CET')

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
week = str(localTz.localize(today - timedelta(weeks=1)))
datasets = ['requirements','requirements_units','requirements_weather','requirements_units_weather']
datasets_dict = {'requirements':'requirements','requirements_units':'requirements_units',
'requirements_weather':'requirements_weather','requirements_units_weather':'requirements_units_weather'}

models = ['Linear','KnnModel','XgbModel','Lstm','Hybrid']
# dataset = 'requirements'
# database = 'requirements_units'

@app.route('/Api/<route>')
def api(route):

	if route =='datasets':
		return pd.DataFrame(datasets)[0].to_dict()
	elif route == 'models':
		return pd.DataFrame(models)[0].to_dict()
	elif route == 'docs':
		return render_template('api.jinja',datasets= datasets,models = models)

@app.route('/')
def home():
	return redirect('Dataset/requirements')

@app.route('/Api')
def api_redict():
	return redirect('Api/docs')

@app.route('/Dataset/<dataset>')
def index(dataset):
	db = DB(datasets_dict[dataset])
	df = db.get_data('*', dataset,f'"index" > "{week}"')
	if 'units' in dataset:
		heatmap = get_heatmap(df.iloc[:,7:-7 if 'cloudCover' in df.columns else -1])
		df = df.drop(axis = 1,columns = df.iloc[:,6:-7 if 'cloudCover' in df.columns else -1])
	else:
		heatmap = None
	return render_template('home.jinja',title = f'Train Data For {dataset} Dataset For The Past 7 Days',
	df=df,get_json = get_json_for_line_fig,candlestick = get_candlesticks(df.SMP),dataset = dataset,heatmap = heatmap)

@app.route('/Correlation/<dataset>')
def corrolations(dataset):
	db = DB(datasets_dict[dataset])
	df = db.get_data('*', dataset,f'"index" > "{week}"')
	if 'units' in dataset:
		df = df.drop(axis = 1,columns =df.iloc[:,6:-7 if 'cloudCover' in df.columns else -1])
	df = df.set_index('SMP').dropna()
	return render_template('correlation.jinja',title = f'Correlation For {dataset} Dataset For The Past 7 Days',
	df=df,get_json = get_json_for_fig_scatter,dataset = dataset)

@app.route('/<name>/<dataset>')
def page_for_ml_model(dataset,name):
	db = DB(datasets_dict[dataset])
	df = db.get_data('"index","SMP","Testing","Inference"',name,f'"index" > "{week}"')
	df['Previous Prediction'] = db.get_data(f'"index","{name}"','infernce')
	
	if 'Lstm' in name:
		hist = db.get_data('*',f'hist_{name}')
		metrics = get_metrics(name,db)
		return render_template('lstm.jinja', title = f'{name} Model {dataset} Dataset Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,
							hist_json = get_json_for_line_scatter(hist,hist.columns,metrics.iloc[0]['best_epoch']),dataset = dataset)
	else:
		metrics = get_metrics(name,db)

		return render_template('model.jinja', title = f'{name} Model {dataset} Dataset Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,dataset = dataset)


@app.route('/prices_api/<dataset>')
def prices_api(dataset):
	try:
		db = DB(datasets_dict[dataset])
		df = pd.DataFrame()
		for model in models:
			df[model] = db.get_data('"index","Inference"',model).dropna()
		return df.reset_index(drop=True).to_dict()
	except:
		return 'No Prediction Possible'

@app.route('/metrics_api/<dataset>/<model>')
def metrics_api(dataset,model):
	db = DB(datasets_dict[dataset])
	try : 
		if model == 'all':
			dict = {}
			for model in models:
				dict[model] = get_metrics(model,db).loc[:,['Train','Validation','Test']].to_dict()
			return dict
		else:
			return get_metrics(model,db).to_dict()
	except:
		return "WRONG"





if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
