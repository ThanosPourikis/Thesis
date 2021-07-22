import flask
from flask import render_template

from utils.database_interface import DB
from utils.utils import get_json_for_line_fig,get_json_for_fig_scatter, get_json_for_line_fig_pred


app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

@app.route('/')
def index():
	db = DB()
	df = db.get_data('dataset','*')
	return render_template('charts.jinja',title = 'Original Data',df=df[-(7*24):],get_json = get_json_for_line_fig,x='Date')

@app.route('/Correlation')
def corrolations():

	db = DB()
	df = db.get_data('dataset','*')
	df = df.iloc[:,df.columns!='Date'].dropna()
	return render_template('charts.jinja',title = 'Correlation',df=df[-(7*24):],get_json = get_json_for_fig_scatter,x='SMP')

@app.route('/Linear')
def Linear_page():
	df = 	db = DB()
	df = db.get_data('linear','*')
	metrics = db.get_data('metrics','linear')


	return render_template('model.jinja', title = 'Linear Model Last 24hours Prediction vs Actual Price',
							chart_json = get_json_for_line_fig(df,'Date',['SMP','Prediction']),
							train_score= train_score,
							validation_score = validation_score)

# @app.route('/KnnR')
# def Knn():
# 	df = get_data_from_csv()

# 	KnnR = KnnModel(data=df)

# 	prediction, train_score, validation_score, test_score, model = KnnR.train()
# 	df['Prediction'] = nan
# 	df[-len(prediction):]['Prediction'] = prediction
# 	# df = df.dropna()

# 	return render_template('model.jinja', title = 'KnnR Model Last 24hours Prediction vs Actual Price',
# 							chart_json = get_json_for_line_fig(df[-48:],'Date',['SMP','Prediction']),
# 							train_score= train_score,
# 							validation_score = validation_score,
# 							test_score = test_score,
# 							model = model)


# @app.route('/XgB')
# def XgB():
# 	df = get_data_from_csv()
# 	xgb = XgbModel(data=df)
# 	prediction,test, train_score, validation_score,test_score,model = xgb.train()
# 	prediction = prediction.set_index('Date').join(df.set_index('Date')['SMP']).reset_index()

# 	return render_template('model.jinja', title = 'XgBoost Regresion Last 24hours Prediction vs Actual Price',
# 							chart_json = get_json_for_line_fig(prediction,'Date',['SMP','Prediction']),
# 							test_json = get_json_for_line_fig(test,'Date',['SMP','Prediction']),
# 							train_score= train_score,
# 							validation_score = validation_score,
# 							test_score = test_score,
# 							model = model)

# @app.route('/Lstm')
# def lstm():
# 	df = get_req()
# 	# df = (pd.read_csv('datasets/requirements.csv').set_index('Date').join(pd.read_csv('datasets/SMP.csv').set_index('Date'))).reset_index()
# 	# df = get_data_from_csv_with_weather()


# 	lstm_model = LstmMVInput(utils.MAE,df) 
# 	y_validation_prediction,hist,lstm = lstm_model.train()
# 	# try:
# 	# 	print('Retraing for best epoch')
# 	# 	lstm_best_model = LstmMVInput(utils.MAE,df,learning_rate=0.01,num_epochs=hist['val'].argmin())
# 	# 	y_validation_prediction,hist,lstm = lstm_best_model.train()
# 	# except:
# 	# 	pass


# 	df['Prediction'] = nan
# 	df[-len(y_validation_prediction):]['Prediction'] = y_validation_prediction
# 	df = df.dropna()

# 	# hist = pd.DataFrame({'hist_train': hist_train ,'hist_val':hist_val})



# 	return render_template('lstm.jinja',title = 'Lstm Last 24hour Prediction vs Actual',
# 												train_score = lstm.loc[0,'train'],
# 												validation_score = lstm.loc[0,'val'],
# 												train_time = lstm.loc[0,'time'],
# 												chart_json = get_json_for_line_fig(df,'Date',['SMP','Prediction']),
# 												hist_json = get_json_for_line_fig(hist,hist.index,['train','val'])
# 												)

# @app.route('/test')
# def test():
# 	return ''

if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
