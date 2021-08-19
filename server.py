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


logging.basicConfig(filename='log.log',level=logging.DEBUG)

def linear():
	db =DB()
	df = db.get_data('*','train_set')
	linear = Linear(data=df)
	linear.train()
	prediction,metrics = linear.get_results()
	db.save_df_to_db(prediction,'Linear')
	utils.save_metrics(metrics,'Linear',db)



def Knn():
	db =DB()
	df = db.get_data('*','train_set')
	knn = KnnModel(data=df)
	knn.train()
	prediction,metrics = knn.get_results()
	db.save_df_to_db(prediction,'knn')

	utils.save_metrics(metrics,'knn',db)

def xgb():
	db =DB()
	df = db.get_data('*','train_set')
	xgb = XgbModel(data=df,booster='gbtree')
	xgb.train()
	prediction,metrics = xgb.get_results()
	db.save_df_to_db(prediction,'xgb')

	utils.save_metrics(metrics,'xgb',db)

def Lstm():
	db =DB()
	df = db.get_data('*','train_set')
	lstm = LstmMVInput(utils.MAE,df,num_epochs=150,batch_size=32,sequence_length=24,name = 'Vanilla')
	lstm.train()
	prediction,metrics,hist,best_epoch = lstm.get_results()
	db.save_df_to_db(hist,'hist_lstm')

	db.save_df_to_db(prediction,'lstm')
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,'lstm',db)

def hybrid_lstm():
	db =DB()
	df = db.get_data('*','train_set')

	if df.isnull().values.any():
		data = df.set_index('Date')
		features = data.loc[:,data.columns != 'SMP'][-(24*8):-24]
		labels = data.loc[:,'SMP'][-(24*8):-24]
		data = data.loc[:,data.columns != 'SMP']
	else:
		data = df.set_index('Date')
		features = data.loc[:,data.columns != 'SMP'][-(24*7):]
		labels = data.loc[:,'SMP'][-(24*7):]
		data = data.loc[:,data.columns != 'SMP']
	

	x_train,y_train = shuffle(features,labels)
	lr = LinearRegression().fit(x_train, y_train)
	
	df['Linear'] = lr.predict(data)


	gsK = GridSearchCV(KNeighborsRegressor(),{'n_neighbors': range(1, 50)}).fit(x_train,y_train)
	df['Knn'] = gsK.predict(data)

	gsX = xgboost.XGBRegressor(learning_rate = 0.1,
								colsample_bytree = 1,
								colsample_bylevel=0.8,
								subsample=0.8,
								n_estimators=49,
								max_depth= 9,
								n_jobs= -1).fit(x_train,y_train)

	df['XGB'] = gsX.predict(data)

	df = df.reset_index()
	df = df.loc[:,['XGB','Knn','Linear','SMP','Date']]

	hybrid_lstm = LstmMVInput(utils.MAE,df,num_epochs=150,batch_size=32,sequence_length=24,name = 'Hybrid')
	hybrid_lstm.train()
	prediction,metrics,hist,best_epoch = hybrid_lstm.get_results()
	db.save_df_to_db(hist,'hist_Hybrid_Lstm')

	db.save_df_to_db(prediction,'Hybrid_Lstm')
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,'Hybrid_Lstm',db)

def save_infernce():
	try:
		db = DB()
		df = pd.DataFrame()
		df['Linear'] = db.get_data('*','Linear').set_index('Date')['Inference'].dropna()
		df['Knn'] = db.get_data('*','Knn').set_index('Date')['Inference'].dropna()
		df['XgB'] = db.get_data('*','XgB').set_index('Date')['Inference'].dropna()
		df['Lstm'] = db.get_data('*','Lstm').set_index('Date')['Inference'].dropna()
		df['Hybrid_Lstm'] = db.get_data('*','Hybrid_Lstm').set_index('Date')['Inference'].dropna()
		db.save_df_to_db(df.reset_index(),'infernce')
	except:
		return 'No Prediction Possible'
		


save_infernce()
update()
threading.Thread(target=linear).start()
threading.Thread(target=Knn).start()
threading.Thread(target=xgb).start()
threading.Thread(target=Lstm).start()
threading.Thread(target=hybrid_lstm).start()

# content = requests.get('http://thanospourikis.pythonanywhere.com/api')
# jsonData = json.loads(content.content)
# infe = pd.DataFrame(jsonData).set_index("Date")

# db =DB()
# df = db.get_data('*','train_set').set_index('Date')[-len(infe):]
# infe = db.get_data('*','infernce').set_index('Date')[-len(infe):]


# for i in infe.columns:
# 	mae = mean_absolute_error(df['SMP'],infe[i])
# 	print(f'{i} : {mae}')
