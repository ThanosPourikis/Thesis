import json

from utils.database_interface import DB
from models.XgB import XgbModel
from sklearn import utils
from models.lstm.LstmMVInput import LstmMVInput
import pandas as pd
from models.KnnModel import KnnModel
from models.Linear import Linear
from utils import utils
import logging
from utils.update_data import update
import threading
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import xgboost
import requests

def train_model(model,model_name,dataset_name):
	db_in =DB(database_in)
	df = db_in.get_data('*',dataset_name)
	model = model(data=df)
	model.train()
	prediction,metrics = model.get_results()
	db_out = DB(dataset_name)
	db_out.save_df_to_db(prediction,model_name)
	utils.save_metrics(metrics,model_name,db_out)

def Lstm(dataset_name):
	db_in =DB(database_in)
	df = db_in.get_data('*',dataset_name)
	lstm = LstmMVInput(utils.MAE,df,num_epochs=20,batch_size=32,sequence_length=24,name = 'Vanilla')
	lstm.train()
	prediction,metrics,hist,best_epoch = lstm.get_results()
	db_out = DB(dataset_name)

	db_out.save_df_to_db(hist,'hist_lstm')
	db_out.save_df_to_db(prediction,'lstm')
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,'lstm',db_out)

def hybrid_lstm(dataset_name):
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


	gsK = KNeighborsRegressor(49,n_jobs=-1).fit(x_train, y_train)
	df['Knn'] = gsK.predict(data)

	gsX = xgboost.XGBRegressor(learning_rate = 0.1,
								colsample_bytree = 1,
								colsample_bylevel=0.8,
								subsample=0.8,
								n_estimators=49,
								max_depth= 9,
								n_jobs= -1).fit(x_train,y_train)

	df['XGB'] = gsX.predict(data)

	df = df.loc[:,['XGB','Knn','Linear','SMP']]

	hybrid_lstm = LstmMVInput(utils.MAE,df,num_epochs=20,batch_size=32,sequence_length=24,name = 'Hybrid')
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

update()
for dataset_name in datasets:
	save_infernce(dataset_name)
	threading.Thread(target=train_model,args = (Linear,'Linear',dataset_name,)).start()
	threading.Thread(target=train_model,args = (KnnModel,'KnnModel',dataset_name,)).start()
	threading.Thread(target=train_model,args = (XgbModel,'XgbModel',dataset_name,)).start()
	threading.Thread(target=Lstm,args = (dataset_name,)).start()
	threading.Thread(target=hybrid_lstm,args = (dataset_name,)).start()

# content = requests.get('http://thanospourikis.pythonanywhere.com/api')
# jsonData = json.loads(content.content)
# infe = pd.DataFrame(jsonData).set_index("Date")

# db =DB(database)
# df = db.get_data('*',dataset).set_index('Date')[-len(infe):]
# infe = db.get_data('*','infernce').set_index('Date')[-len(infe):]


# for i in infe.columns:
# 	mae = mean_absolute_error(df['SMP'],infe[i])
# 	print(f'{i} : {mae}')

