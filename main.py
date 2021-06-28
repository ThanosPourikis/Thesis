from data.get_SMP_data import get_SMP_data
from math import nan
from models.XgB import XgbModel
import numpy as np
from sklearn import utils
from models.LstmMVInput import LstmMVInput
import pandas as pd
from models.KnnModel import KnnModel
from models.Linear import Linear
from models.Lstm_model import LSTM
import flask
from flask import render_template
from utils import utils
from utils.utils import get_data, get_json_for_line_fig,get_json_for_fig_scatter,get_data_from_csv

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config['Debug'] = True


@app.route('/')
def index():
	df = get_data_from_csv()
	return render_template('charts.jinja',title = 'Original Data',df=df,get_json = get_json_for_line_fig,y='Date')

@app.route('/Correlation')
def corrolations():

	df = get_data_from_csv()
	df = df.loc[:,df.columns!='Date'].dropna()
	return render_template('charts.jinja',title = 'Correlation',df=df,y='SMP',get_json = get_json_for_fig_scatter,

							)

@app.route('/Linear')
def Linear_page():
	df = get_data_from_csv()

	linear = Linear(data=df)
	

	prediction, train_score, validation_score = linear.run()

	return render_template('model.jinja', title = 'Linear Model Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(prediction,'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/KnnR')
def Knn():
	df = get_data_from_csv()

	KnnR = KnnModel(data=df)

	prediction, train_score, validation_score = KnnR.run()
	df['Prediction'] = nan
	df[-len(prediction):]['Prediction'] = prediction
	df = df.dropna()

	return render_template('model.jinja', title = 'KnnR Model Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df,'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)


@app.route('/XgB')
def XgB():
	df = get_data_from_csv()
	xgb = XgbModel(data=df)
	prediction, train_score, validation_score = xgb.run()
	
	return render_template('model.jinja', title = 'XgBoost Regresion Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(prediction,'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

@app.route('/Lstm')
def lstm():
	df = get_data_from_csv()

	try:
		y_validation_prediction = np.array(get_data('lstmPredictiona','*')).flatten().tolist()
		hist = get_data('lstm_hist','*')
		lstm = get_data('lstm_metrics','*')
	except :
		lstm_model = LstmMVInput(utils.MAE,df,learning_rate=0.01)
		y_validation_prediction,hist,lstm = lstm_model.run()

	df['Prediction'] = nan
	df[-len(y_validation_prediction[:-24]):]['Prediction'] = y_validation_prediction[:-24]
	df = df.dropna()

	# hist = pd.DataFrame({'hist_train': hist_train ,'hist_val':hist_val})



	return render_template('lstm.jinja',title = 'Lstm Last 24hour Prediction vs Actual',
												train_score = lstm.loc[0,'train'],
												validation_score = lstm.loc[0,'val'],
												train_time = lstm.loc[0,'time'],
												chart_json = get_json_for_line_fig(df,'Date',['SMP','Prediction']),
												hist_json = get_json_for_line_fig(hist,hist.index,['train','val'])
												)

@app.route('/test')
def test():
	return ''

if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
