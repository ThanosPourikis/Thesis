import logging
import time
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


class KnnModel:
	def __init__(self, data, n_neighbors_parameters = 50, validation_size =0.2):
		data = data.set_index('Date')
		# data = data.loc[:,['Renewables','Non-Dispatcheble','Man_Hydro','SMP']]
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
		self.params = {'leaf_size': [10], 'n_neighbors': [5]}
		
		

	def train(self):
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels, random_state=96,
																	test_size=self.validation_size, shuffle=True)

		logging.info("Training ... ")
		start_time = time.time()
		self.model = KNeighborsRegressor(n_neighbors=5)
		self.model.fit(self.x_train, self.y_train)
		logging.info(f'Time:{time.time() - start_time}')

	def get_results(self):
		train_error = mean_absolute_error(self.y_train,self.model.predict(self.x_train))
		validate_error = mean_absolute_error(self.y_validate,self.model.predict(self.x_validate))
		self.model.fit(self.x_validate,self.y_validate)
		pred = self.model.predict(self.test.loc[:,self.test.columns != 'SMP'])
		test_error = mean_absolute_error(self.test.loc[:,'SMP'],pred)
		self.test['Prediction'] = pred
		try:
			self.inference['Inference'] = self.model.predict(self.inference.loc[:,self.inference.columns != 'SMP'])
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			return self.test.iloc[:,-3:].reset_index(),train_error,validate_error,test_error

		except:
			return self.test.iloc[:,-2:].reset_index(),train_error,validate_error,test_error