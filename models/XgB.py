import logging
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from models.utils import get_metrics_df



class XgbModel:
	def __init__(self,data,booster,validation_size =0.2):
		data = data.set_index('Date')
		if data.isnull().values.any():
			self.inference = data[-24:]
			self.test = data[-(8*24):-24]
			data = data[:-(8*24)]
		else:
			self.test = data[-(7*24):]
			data = data[:-(7*24)]

		self.validation_size = validation_size
		self.features = data.loc[:,data.columns!='SMP'].reset_index(drop=True)
		self.labels = (data.loc[:,data.columns=='SMP']).reset_index(drop=True)
		self.data = data
		self.booster = booster
		self.parameters = {
		"learning_rate": [0.09],
		"max_depth": [10],
		"n_estimators": [100],
		"colsample_bytree": [0.8],
		"random_state": [96],
		}
		self.parameters = {
			"learning_rate": [0.1,0.5],
			"max_depth": [9],
			"n_estimators": [49],
			'subsample': [0.8],
			"colsample_bytree": [1],
			'colsample_bylevel' : [0.8],
			"random_state": [96],
		}

	def train(self):

		self.labels = self.labels.reset_index(drop = True).dropna()
		self.features = (self.features).loc[:,self.features.columns!='Date'][:len(self.labels)].dropna()
		self.model = xgboost.XGBRegressor(learning_rate = 0.09,colsample_bytree = 0.8, n_estimators=49,max_depth= 8,booster = self.booster,n_jobs= -1)
		# cs = GridSearchCV(model, self.parameters)
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels,shuffle=True,random_state=96,
																	test_size=self.validation_size)

		logging.info('Fitting Model')

		self.model.fit(self.x_train,self.y_train)
	
		# cs.fit(x_train,y_train)
		# model=xgboost.XGBRegressor(**cs.best_params_)
	def get_results(self):
		# self.model.fit(self.x_validate,self.y_validate)

		pred = self.model.predict(self.test.loc[:,self.test.columns != 'SMP'])
		metrics = get_metrics_df(
			self.y_train,self.model.predict(self.x_train),
			self.y_validate,self.model.predict(self.x_validate),
			self.test.loc[:,'SMP'],pred)
		self.test['Prediction'] = pred
		try:
			self.inference['Inference'] = self.model.predict(self.inference.loc[:,self.inference.columns != 'SMP'])
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			return self.test.iloc[:,-3:].reset_index(),metrics
		except:
			return self.test.iloc[:,-2:].reset_index(),metrics