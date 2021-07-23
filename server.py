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

class Server:
	def __init__(self):
		self.db =DB()
		self.df = self.db.get_data('*','train_set')
		self.today = pd.to_datetime(date.today()) #+ timedelta(days= 1)
	def linear(self):
		linear = Linear(data=self.df)
		linear.train()
		prediction,train_error,validate_error,test_error = linear.get_res()
		self.db.save_df_to_db(prediction,'linear')

		metrics = pd.DataFrame({
			'train_error':train_error,
			'validate_error':validate_error,
			'test_error':test_error},index=[self.today])
		utils.save_metrics(metrics,'linear',self.db)
		metrics = utils.get_metrics('linear',self.db)



	def Knn(self):
		knn = KnnModel(data=self.df)
		knn.train()
		prediction,train_error,validate_error,test_error,best_params_ = knn.get_res()
		self.db.save_df_to_db(prediction,'knn')

		metrics = pd.DataFrame({
			'train_error':train_error,
			'validate_error':validate_error,
			'test_error':test_error,
			'best_params_':best_params_['n_neighbors']},index=[self.today])
		utils.save_metrics(metrics,'knn',self.db)

	def xgb(self):
		xgb = XgbModel(data=self.df)
		xgb.train()
		prediction,train_error,validate_error,test_error = xgb.get_res()
		self.db.save_df_to_db(prediction,'xgb')

		metrics = pd.DataFrame({
			'train_error':train_error,
			'validate_error':validate_error,
			'test_error':test_error},index=[self.today])
		utils.save_metrics(metrics,'xgb',self.db)

	def lstm(self):
		lstm = LstmMVInput(utils.MAE,self.df,num_epochs=150,batch_size=32,sequence_length=24)
		lstm.train()
		prediction,train_error,validate_error,test_error,hist = lstm.get_res()
		self.db.save_df_to_db(hist,'hist_lstm')

		self.db.save_df_to_db(prediction,'lstm')

		metrics = pd.DataFrame({
			'train_error':train_error,
			'validate_error':validate_error,
			'test_error':test_error},index=[self.today])
		utils.save_metrics(metrics,'lstm',self.db)


update()
server = Server()
server.linear()
server.Knn()
server.xgb()
server.lstm()

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
