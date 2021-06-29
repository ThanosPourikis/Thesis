import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error



class XgbModel:
	def __init__(self,data,validation_size =0.2):
		self.validation_size = validation_size
		self.features = data[:-24].reset_index(drop=True)
		self.labels = (data.loc[:,data.columns=='SMP'][24:]).reset_index(drop=True)
		self.date = data.loc[:,data.columns=='Date']
		self.data = data
		# self.parameters = {
		# "learning_rate": [0.09],
		# "max_depth": [3, 4, 5, 6],
		# "n_estimators": [100, 1000],
		# "colsample_bytree": [0.8],
		# "random_state": [96],
		# }

	def run(self):
		self.features = (self.features).loc[:,self.features.columns!='Date'].dropna()
		xgboost.set_config(verbosity = 2)
		model = xgboost.XGBRegressor(learning_rate = 0.3,colsample_bytree = 0.8)
		# cs = GridSearchCV(model, self.parameters)
		x_train, x_validate, y_train, y_validate = train_test_split(self.features, self.labels,shuffle=True,random_state=96,
																	test_size=self.validation_size)

		print('Fitting Model')

		model.fit(x_train,y_train)
		
		print('Fitted Model')
		train_error = mean_absolute_error(y_train, model.predict(x_train))
		validate_error = mean_absolute_error(y_validate, model.predict(x_validate))
		x_validate['Prediction'] = model.predict(x_validate)
		x_validate = x_validate.join(self.date)
		x_validate = x_validate.sort_index()
		return x_validate,train_error,validate_error,model

