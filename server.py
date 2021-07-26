from utils.database_interface import DB
from models.XgB import XgbModel
from sklearn import utils
from models.lstm.LstmMVInput import LstmMVInput
import pandas as pd
from models.KnnModel import KnnModel
from models.Linear import Linear
from utils import utils
from datetime import date
import logging
from utils.update_data import update
import threading
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import xgboost


logging.basicConfig(filename='log.log',level=logging.DEBUG)

def linear():
	db =DB()
	df = db.get_data('*','train_set')
	linear = Linear(data=df)
	linear.train()
	prediction,train_error,validate_error,test_error = linear.get_results()
	db.save_df_to_db(prediction,'linear')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'linear',db)



def Knn():
	db =DB()
	df = db.get_data('*','train_set')
	knn = KnnModel(data=df)
	knn.train()
	prediction,train_error,validate_error,test_error,best_params_ = knn.get_results()
	db.save_df_to_db(prediction,'knn')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error,
		'best_params_':best_params_['n_neighbors']},index=[today])
	utils.save_metrics(metrics,'knn',db)

def xgb():
	db =DB()
	df = db.get_data('*','train_set')
	xgb = XgbModel(data=df)
	xgb.train()
	prediction,train_error,validate_error,test_error = xgb.get_results()
	db.save_df_to_db(prediction,'xgb')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'xgb',db)

def Lstm():
	db =DB()
	df = db.get_data('*','train_set')
	lstm = LstmMVInput(utils.MAE,df,num_epochs=50,batch_size=32,sequence_length=24)
	lstm.train()
	prediction,train_error,validate_error,test_error,hist = lstm.get_results()
	db.save_df_to_db(hist,'hist_lstm')

	db.save_df_to_db(prediction,'lstm')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'lstm',db)

def hybrid_lstm():
	db =DB()
	df = db.get_data('*','train_set')

	if df.isnull().values.any():
		data = df.set_index('Date')
		features = data.loc[:,data.columns != 'SMP'][:-24]
		labels = data.loc[:,'SMP'][:-24]
		data = data.loc[:,data.columns != 'SMP']
	else:
		data = df.set_index('Date')
		features = data.loc[:,data.columns != 'SMP']
		labels = data.loc[:,'SMP']
		data = data.loc[:,data.columns != 'SMP']
	

	x_train,y_train = shuffle(features,labels)
	lr = LinearRegression().fit(x_train, y_train)
	
	df['Linear'] = lr.predict(data)


	gsK = GridSearchCV(KNeighborsRegressor(),{'n_neighbors': range(1, 50)}).fit(x_train,y_train)
	df['Knn'] = gsK.predict(data)

	gsX = xgboost.XGBRegressor(learning_rate = 0.09,colsample_bytree = 0.8, n_estimators=100,max_depth= 8).fit(x_train,y_train)

	df['XGB'] = gsX.predict(data)

	hybrid_lstm = LstmMVInput(utils.MAE,df,num_epochs=50,batch_size=32,sequence_length=24)
	hybrid_lstm.train()
	prediction,train_error,validate_error,test_error,hist = hybrid_lstm.get_results()
	db.save_df_to_db(hist,'hist_Hybrid_Lstm')

	db.save_df_to_db(prediction,'Hybrid_Lstm')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'Hybrid_Lstm',db)


today = pd.to_datetime(date.today()) #+ timedelta(days= 1)

# update()
# threading.Thread(target=linear).start()
# threading.Thread(target=Knn).start()
# threading.Thread(target=xgb).start()
# threading.Thread(target=Lstm).start()
threading.Thread(target=hybrid_lstm).start()