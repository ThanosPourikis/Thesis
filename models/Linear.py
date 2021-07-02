
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from utils.utils import mean_absolute_error


class Linear:
	def __init__(self, data, validation_size=0.2):
		# self.features = data.loc[:,data.columns!='SMP']
		self.features = data.loc[:,data.columns!='SMP'].reset_index(drop=True)
		self.labels = (data.loc[:,data.columns=='SMP']).reset_index(drop=True)
		self.date = data.loc[:,data.columns=='Date']
		self.validation_size = validation_size
		self.date = data.loc[:,data.columns=='Date']
		self.data = data

	def run(self):
		
		self.labels = self.labels.reset_index(drop = True).dropna()
		self.features = (self.features).loc[:,self.features.columns!='Date'][:len(self.labels)].dropna()
		
		x_train, x_validate, y_train, y_validate = train_test_split(self.features, self.labels, random_state=96,
																	test_size=self.validation_size, shuffle=True)

													
		lr = LinearRegression().fit(x_train, y_train)
		

		train_error = mean_absolute_error(y_train, lr.predict(x_train))
		validate_error = mean_absolute_error(y_validate, lr.predict(x_validate))
		x_validate['Prediction'] = lr.predict(x_validate)
		x_validate = x_validate.join(self.date).join(self.labels)
		x_validate = x_validate.sort_index()
		return x_validate,train_error,validate_error