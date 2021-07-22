import time
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


class KnnModel:
	def __init__(self, data, n_neighbors_parameters = 50, validation_size =0.2):
		self.features = data.loc[:,data.columns!='SMP'].reset_index(drop=True)
		self.labels = (data.loc[:,data.columns=='SMP']).reset_index(drop=True)
		self.date = data.loc[:,data.columns=='Date']
		self.validation_size = validation_size
		self.data = data
		self.n_neighbors_parameters = {'n_neighbors': range(1, n_neighbors_parameters)}

	def train(self):
		self.labels = self.labels.reset_index(drop = True).dropna()
		self.features = (self.features).loc[:,self.features.columns!='Date'][:len(self.labels)].dropna()

		x_train, x_validate, y_train, y_validate = train_test_split(self.features[:-24], self.labels[:-24], random_state=96,
																	test_size=self.validation_size, shuffle=True)


		
		print("Training ... ")
		start_time = time.time()
		gs = GridSearchCV(KNeighborsRegressor(), self.n_neighbors_parameters)
		gs.fit(x_train, y_train)
		print(f'Time:{time.time() - start_time}')
		print(gs.best_params_)
		self.model = gs 
		return gs.predict(self.features[-48:]), mean_absolute_error(y_train, gs.predict(x_train)), mean_absolute_error(y_validate, gs.predict(x_validate)),mean_absolute_error(self.labels[-24:], gs.predict(self.features[-24:])),gs.best_params_
	
	def predict(self,df):
		if self.model == None:
			print('Pleaze Train Model')
		else:
			return self.model.predict(df)
	