import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd



class XgbModel:
	def __init__(self,data,validation_size =0.2):
		self.validation_size = validation_size
		self.features = data.loc[:,data.columns!='SMP'].reset_index(drop=True)
		self.labels = (data.loc[:,data.columns=='SMP']).reset_index(drop=True)
		self.test = data.loc[:,data.columns!='SMP'][-24:]
		self.date = data.loc[:,data.columns=='Date']
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
		xgboost.set_config(verbosity = 2)
		model = xgboost.XGBRegressor(learning_rate = 0.09,colsample_bytree = 0.8, n_estimators=100,max_depth= 8)
		# cs = GridSearchCV(model, self.parameters)
		x_train, x_validate, y_train, y_validate = train_test_split(self.features[:-24], self.labels[:-24],shuffle=True,random_state=96,
																	test_size=self.validation_size)

		print('Fitting Model')

		model.fit(x_train,y_train)

		# cs.fit(x_train,y_train)
		# model=xgboost.XGBRegressor(**cs.best_params_)
		print('Model Fitted')
		model.save_model('xgbR.model')
		train_error = mean_absolute_error(y_train, model.predict(x_train))
		validate_error = mean_absolute_error(y_validate, model.predict(x_validate))
		test_error = mean_absolute_error(self.labels[-24:],model.predict(self.features[-24:]))
		x_validate['Prediction'] = model.predict(x_validate)
		x_validate = x_validate.join(self.date)
		x_validate = x_validate.sort_index()
		x_test = self.features[-24:].join(self.labels[-24:]).join(self.date)
		x_test['Prediction'] = model.predict(self.features[-24:])
		
		return x_validate,x_test,train_error,validate_error,test_error,model

