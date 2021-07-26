import logging
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd



class XgbModel:
	def __init__(self,data,validation_size =0.2):
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
		self.parameters = {
		"learning_rate": [0.09],
		"max_depth": [10],
		"n_estimators": [100],
		"colsample_bytree": [0.8],
		"random_state": [96],
		}

	def train(self):

		self.labels = self.labels.reset_index(drop = True).dropna()
		self.features = (self.features).loc[:,self.features.columns!='Date'][:len(self.labels)].dropna()
		self.model = xgboost.XGBRegressor(learning_rate = 0.09,colsample_bytree = 0.8, n_estimators=100,max_depth= 8)
		# cs = GridSearchCV(model, self.parameters)
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels,shuffle=True,random_state=96,
																	test_size=self.validation_size)

		logging.info('Fitting Model')

		self.model.fit(self.x_train,self.y_train)
	
		# cs.fit(x_train,y_train)
		# model=xgboost.XGBRegressor(**cs.best_params_)
	def get_results(self):
		train_error = mean_absolute_error(self.y_train, self.model.predict(self.x_train))
		validate_error = mean_absolute_error(self.y_validate, self.model.predict(self.x_validate))

		pred = self.model.predict(self.test.loc[:,self.test.columns != 'SMP'])
		test_error = mean_absolute_error(pred,self.test['SMP'])
		self.test['Prediction'] = pred
		try:
			self.inference['Inference'] = self.model.predict(self.inference.loc[:,self.inference.columns != 'SMP'])
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			return self.test.iloc[:,-3:].reset_index(),train_error,validate_error,test_error
		except:
			return self.test.iloc[:,-2:].reset_index(),train_error,validate_error,test_error