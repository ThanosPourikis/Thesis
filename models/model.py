
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from models.utils import get_metrics_df


def get_model_results(data,params,name,model, validation_size=0.2):
	# features = data.loc[:,data.columns!='SMP']
	# data = data.set_index('Date')
	if data.isnull().values.any():
		inference = data[-24:]
		test = data[-(8*24):-24]
		data = data[:-(8*24)]
	else:
		test = data[-(7*24):]
		data = data[:-(7*24)]

	name = name
	features = data.loc[:,data.columns!='SMP']
	labels = (data.loc[:,'SMP'])
	model = model(**params)

	x_train, x_validate, y_train, y_validate = train_test_split(features, labels, random_state=96,
																test_size=validation_size, shuffle=True)

	logging.info(f'Fitting {name} Model')

	model.fit(x_train,y_train)
	
	pred = model.predict(test.loc[:,test.columns != 'SMP'])

	metrics = get_metrics_df(
		y_train,model.predict(x_train),
		y_validate,model.predict(x_validate),
		test.loc[:,'SMP'],pred)
	test['Prediction'] = pred
	try:
		inference['Inference'] = model.predict(inference.loc[:,inference.columns != 'SMP'])
		test = pd.concat([test,inference['Inference']],axis=1)
		return test.iloc[:,-3:],metrics
	except:
		return test.iloc[:,-2:],metrics

		