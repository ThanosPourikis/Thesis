import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from utils.utils import mean_absolute_error


class Linear:
	def __init__(self, data, validation_size=0.2):
		self.features = data.loc[:,data.columns!='SMP']
		self.labels = data.loc[:,data.columns=='SMP']
		self.validation_size = validation_size

	def run_linear(self):


		self.features = (self.features).loc[:,self.features.columns!='Date'].dropna()

		x_train, x_validate, y_train, y_validate = train_test_split(self.features, self.labels, random_state=96,
																	test_size=self.validation_size, shuffle=True)
		lr = LinearRegression().fit(x_train, y_train)
		return pd.DataFrame(lr.predict(self.features)), mean_absolute_error(y_train, lr.predict(x_train)), mean_absolute_error(y_validate, lr.predict(x_validate))
