import json
import yaml
from utils.database_interface import DB
from sklearn import utils
from models.lstm.LstmMVInput import LstmMVInput
import pandas as pd
from models.model import get_model_results
from utils import utils
import logging
from utils.update_data import update
import threading
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import requests

def train_model(model,model_name,dataset_name,params):
	db_in =DB(database_in)
	df = db_in.get_data('*',dataset_name)
	prediction,metrics = get_model_results(df,params,model_name,model)
	db_out = DB(dataset_name)
	db_out.save_df_to_db(prediction,model_name)
	utils.save_metrics(metrics,model_name,db_out)

def Lstm(dataset_name,params):
	db_in =DB(database_in)
	df = db_in.get_data('*',dataset_name)
	lstm = LstmMVInput(utils.MAE,df,name = f'Vanilla {dataset_name}',**params['lstm_params'])
	lstm.train()
	prediction,metrics,hist,best_epoch = lstm.get_results()
	db_out = DB(dataset_name)

	db_out.save_df_to_db(hist,'hist_lstm')
	db_out.save_df_to_db(prediction,'lstm')
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,'lstm',db_out)

def hybrid_lstm(dataset_name,params):
	db_in =DB(database_in)
	df = db_in.get_data('*',dataset_name)

	if df.isnull().values.any():
		data = df.copy()
		features = data.loc[:,data.columns != 'SMP'][:-(24*8)]
		labels = data.loc[:,'SMP'][:-(24*8)]
		data = data.loc[:,data.columns != 'SMP']
	else:
		data = df.copy()
		features = data.loc[:,data.columns != 'SMP'][:-(24*7)]
		labels = data.loc[:,'SMP'][:-(24*7)]
		data = data.loc[:,data.columns != 'SMP']
	

	x_train,y_train = shuffle(features,labels)
	lr = LinearRegression().fit(x_train, y_train)
	
	df['Linear'] = lr.predict(data)


	gsK = KNeighborsRegressor(**params['knn_params'],n_jobs=-1).fit(x_train, y_train)
	df['Knn'] = gsK.predict(data)

	gsX = XGBRegressor(**params['xgb_params']).fit(x_train,y_train)

	df['XGB'] = gsX.predict(data)

	df = df.loc[:,['XGB','Knn','Linear','SMP']]

	hybrid_lstm = LstmMVInput(utils.MAE,df,f'Hybrid {dataset_name}',**params['lstm_params'])
	hybrid_lstm.train()
	prediction,metrics,hist,best_epoch = hybrid_lstm.get_results()
	db_out = DB(dataset_name)

	db_out.save_df_to_db(hist,'hist_Hybrid_Lstm')

	db_out.save_df_to_db(prediction,'Hybrid_Lstm')
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,'Hybrid_Lstm',db_out)

def save_infernce(dataset_name):
	try:
		db = DB(dataset_name)
		df = pd.DataFrame()
		df['Linear'] = db.get_data('"index","Inference"','Linear').dropna()
		df['KnnModel'] = db.get_data('"index","Inference"','KnnModel').dropna()
		df['XgbModel'] = db.get_data('"index","Inference"','XgbModel').dropna()
		df['Lstm'] = db.get_data('"index","Inference"','Lstm').dropna()
		df['Hybrid_Lstm'] = db.get_data('"index","Inference"','Hybrid_Lstm').dropna()
		try:
			df = pd.concat([db.get_data('*','infernce'),df])
			df = df.reset_index().drop_duplicates(subset='index').set_index('index')
		except:
			pass
		db.save_df_to_db(df,'infernce')
	except:
		return 'No Prediction Possible'
		
logging.basicConfig(filename='log.log',level=logging.DEBUG)
datasets = ['requirements','requirements_units','requirements_weather','requirements_units_weather']
database_in = 'dataset'
try: 
	with open ('yaml.yaml', 'r') as file:
		params = yaml.safe_load(file)
except Exception as e:
	print(e)
	
update()
for dataset_name in datasets:
	save_infernce(dataset_name)
	threading.Thread(target=train_model,args = (LinearRegression,'Linear',dataset_name,params['linear_params'],)).start()
	threading.Thread(target=train_model,args = (KNeighborsRegressor,'KnnModel',dataset_name,params['knn_params'],)).start()
	threading.Thread(target=train_model,args = (XGBRegressor,'XgbModel',dataset_name,params['xgb_params'],)).start()
	threading.Thread(target=Lstm,args = (dataset_name,params,)).start()
	threading.Thread(target=hybrid_lstm,args = (dataset_name,params,)).start()

# content = requests.get('http://thanospourikis.pythonanywhere.com/api')
# jsonData = json.loads(content.content)
# infe = pd.DataFrame(jsonData).set_index("Date")

# db =DB(database)
# df = db.get_data('*',dataset).set_index('Date')[-len(infe):]
# infe = db.get_data('*','infernce').set_index('Date')[-len(infe):]


# for i in infe.columns:
# 	mae = mean_absolute_error(df['SMP'],infe[i])
# 	print(f'{i} : {mae}')

