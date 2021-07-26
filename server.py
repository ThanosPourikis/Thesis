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

logging.basicConfig(filename='log.log',level=logging.DEBUG)

def linear(df,db):
	linear = Linear(data=df)
	linear.train()
	prediction,train_error,validate_error,test_error = linear.get_res()
	db.save_df_to_db(prediction,'linear')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'linear',db)
	metrics = utils.get_metrics('linear',db)



def Knn(df,db):
	knn = KnnModel(data=df)
	knn.train()
	prediction,train_error,validate_error,test_error,best_params_ = knn.get_res()
	db.save_df_to_db(prediction,'knn')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error,
		'best_params_':best_params_['n_neighbors']},index=[today])
	utils.save_metrics(metrics,'knn',db)

def xgb(df,db):
	xgb = XgbModel(data=df)
	xgb.train()
	prediction,train_error,validate_error,test_error = xgb.get_res()
	db.save_df_to_db(prediction,'xgb')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'xgb',db)

def lstm(df,db):
	lstm = LstmMVInput(utils.MAE,df,num_epochs=50,batch_size=32,sequence_length=24)
	lstm.train()
	prediction,train_error,validate_error,test_error,hist = lstm.get_res()
	db.save_df_to_db(hist,'hist_lstm')

	db.save_df_to_db(prediction,'lstm')

	metrics = pd.DataFrame({
		'train_error':train_error,
		'validate_error':validate_error,
		'test_error':test_error},index=[today])
	utils.save_metrics(metrics,'lstm',db)

db =DB()
df = db.get_data('*','train_set')
today = pd.to_datetime(date.today()) #+ timedelta(days= 1)

update()
linear(df,db)
Knn(df,db)
xgb(df,db)
lstm(df,db)

# ## Debug ###
# def debug(metrics,prediction):
# 	train_score= metrics['train_score'],
# 	validation_score = metrics['validation_score'],
# 	test_score = metrics['test_score']

# 	if (prediction.columns == 'Inference').any():
# 		fig = px.line(prediction,x='Date',y=['Prediction','SMP','Inference'])
# 	else:
# 		fig = px.line(prediction,x='Date',y=['Prediction','SMP'])

# 	fig = fig.update_xaxes(rangeslider_visible=True)
# 	fig.show()
