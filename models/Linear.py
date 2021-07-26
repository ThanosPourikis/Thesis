
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class Linear:
	def __init__(self, data, validation_size=0.2):
		# self.features = data.loc[:,data.columns!='SMP']
		data = data.set_index('Date')
		if data.isnull().values.any():
			self.inference = data[-24:]
			self.test = data[-(8*24):-24]
			data = data[:-(8*24)]
		else:
			self.test = data[-(7*24):]
			data = data[:-(7*24)]


		self.features = data.loc[:,data.columns!='SMP']
		self.labels = (data.loc[:,data.columns=='SMP'])
		self.validation_size = validation_size


	def train(self):
		logging.info('Fitting Linear Model')
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels, random_state=96,
																	test_size=self.validation_size, shuffle=True)

		self.lr = LinearRegression().fit(self.x_train, self.y_train)
		
	def get_results(self):
		train_error = mean_absolute_error(self.y_train, self.lr.predict(self.x_train))
		validate_error = mean_absolute_error(self.y_validate, self.lr.predict(self.x_validate))

		pred = self.lr.predict(self.test.loc[:,self.test.columns != 'SMP'])
		test_error = mean_absolute_error(pred,self.test['SMP'])
		self.test['Prediction'] = pred
		try:
			self.inference['Inference'] = self.lr.predict(self.inference.loc[:,self.inference.columns != 'SMP'])
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			return self.test.iloc[:,-3:].reset_index(),train_error,validate_error,test_error
		except:
			return self.test.iloc[:,-2:].reset_index(),train_error,validate_error,test_error

		