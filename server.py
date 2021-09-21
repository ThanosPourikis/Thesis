import yaml
from utils.database_interface import DB
from sklearn import utils
from models.lstm.LstmMVInput import LstmMVInput
import pandas as pd
from models.model import get_model_results
from utils import utils
import logging
from utils.update_data import update
import threading
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from models.lstm.Lstm_model import LSTM
from models.lstm.Hybrid_Lstm_model import Hybrid_LSTM
from sklearn.model_selection import train_test_split
from models.utils import get_metrics_df


def train_model(model,model_name,df,dataset_name,params):
	prediction,metrics = get_model_results(df,params[dataset_name],model_name,model)
	db_out = DB(dataset_name)
	db_out.save_df_to_db(prediction,model_name)
	utils.save_metrics(metrics,model_name,db_out)

def Lstm(LSTM,name,df,dataset_name,params):
	lstm = LstmMVInput(utils.MAE,df,name = f'{name} {dataset_name}',LSTM = LSTM,**params)
	lstm.train()
	prediction,metrics,hist,best_epoch = lstm.get_results()
	db_out = DB(dataset_name)

	db_out.save_df_to_db(hist,f'hist_{name}')
	db_out.save_df_to_db(prediction,name)
	metrics['best_epoch'] = best_epoch
	utils.save_metrics(metrics,name,db_out)

def save_infernce(dataset_name):
	try:
		db = DB(dataset_name)
		df = pd.DataFrame()
		df['Linear'] = db.get_data('"index","Inference"','Linear').dropna()
		df['KnnModel'] = db.get_data('"index","Inference"','KnnModel').dropna()
		df['XgbModel'] = db.get_data('"index","Inference"','XgbModel').dropna()
		df['Lstm'] = db.get_data('"index","Inference"','Lstm').dropna()
		df['Hybrid_Lstm'] = db.get_data('"index","Inference"','Hybrid_Lstm').dropna()
		try:
			df = pd.concat([db.get_data('*','infernce'),df])
			df = df.reset_index().drop_duplicates(subset='index').set_index('index')
		except:
			pass
		db.save_df_to_db(df,'infernce')
	except:
		return 'No Prediction Possible'
		
logging.basicConfig(filename='log.log',level=logging.DEBUG)
datasets = ['requirements','requirements_units','requirements_weather','requirements_units_weather']
database_in = 'dataset'
try: 
	with open ('yaml.yaml', 'r') as file:
		params = yaml.safe_load(file)
except Exception as e:
	print(e)

# update()
db_in =DB(database_in)
threads = []
for dataset_name in datasets:
	# save_infernce(dataset_name)
	prediction = db_in.get_data('*',dataset_name)
	prediction.insert(prediction.shape[1]-1,'lag_24',prediction['SMP'].shift(24))
	prediction = prediction[prediction['lag_24'].notna()]
	threading.Thread(target=train_model,args = (LinearRegression,'Linear',prediction,dataset_name,params['linear_params'],)).start()
	threading.Thread(target=train_model,args = (KNeighborsRegressor,'KnnModel',prediction,dataset_name,params['knn_params'],)).start()
	threading.Thread(target=train_model,args = (XGBRegressor,'XgbModel',prediction,dataset_name,params['xgb_params'],)).start()
	threads.append(threading.Thread(target=Lstm,args = (LSTM,'Lstm',prediction,dataset_name,params['Lstm_params'],)))

for i in threads:
	i.start()

for i in threads:
	i.join()

print("Help")
for i in datasets:
	db=DB(i)
	linear =db.get_data('*','Linear')
	knn = db.get_data('*','KnnModel')
	xgb = db.get_data('*','XgbModel')
	lstm = db.get_data('*','Lstm')

	smp = linear['SMP']

	train = pd.concat([linear['Training'].dropna(),knn['Training'].dropna(),xgb['Training'].dropna(),lstm['Training'].dropna()],
	axis=1).mean(axis=1)
	val = pd.concat([linear['Validation'].dropna(),knn['Validation'].dropna(),xgb['Validation'].dropna(),lstm['Validation'].dropna()],
	axis=1).mean(axis=1)
	test = pd.concat([linear['Testing'].dropna(),knn['Testing'].dropna(),xgb['Testing'].dropna(),lstm['Testing'].dropna()],
	axis=1).mean(axis=1)
	inf = pd.concat([linear['Inference'].dropna(),knn['Inference'].dropna(),xgb['Inference'].dropna(),lstm['Inference'].dropna()],
	axis=1).mean(axis=1)
	prediction = pd.concat([smp,train,val,test,inf],axis=1)
	prediction.columns=['SMP','Training','Validation','Testing','Inference']

	metrics = get_metrics_df(prediction.loc[:,['SMP','Training']].dropna()['SMP'],prediction['Training'].dropna(),
	prediction.loc[:,['SMP','Validation']].dropna()['SMP'],prediction['Validation'].dropna(),
	prediction.loc[:,['SMP','Testing']].dropna()['SMP'],prediction['Testing'].dropna())

	prediction,metrics
	db_out = DB(i)

	db_out.save_df_to_db(prediction,'Hybrid')
	utils.save_metrics(metrics,'Hybrid',db_out)