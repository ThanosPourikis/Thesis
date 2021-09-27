import sqlalchemy as sq
import pandas as pd
from pytz import timezone
from datetime import datetime
# import config
localTz = timezone('CET')
sqlite = 'sqlite:///{}.db'
# mysql = f'mysql+mysqldb://{config.USERNAME}:{config.PASSWORD}@{config.HOSTNAME}/{config.DATABASENAME}'
class DB:
	def __init__(self,database):
		self.engine = sq.create_engine(sqlite.format(database),)
		self.connection = self.engine.connect()
	def save_df_to_db(self, dataframe, df_name):
		try:
			dataframe.index = [str(i) for i in dataframe.index]
		except:
			pass
		dataframe.to_sql(df_name, self.connection, if_exists='replace')
	
	def save_inference_to_DB():
		pass

	def save_metrics(self,metrics,model):
		try:
			# metrics = pd.concat([metrics,db.get_data('*',f'metrics_{model}')])
			# metrics = metrics.reset_index().drop_duplicates(subset ='index',keep='first').set_index('index')
			self.save_df_to_db(metrics,f'metrics_{model}')
		except:
			metrics = metrics.reset_index().drop_duplicates(subset ='index',keep='first').set_index('index')
			self.save_df_to_db(metrics,f'metrics_{model}')

	def get_data(self, columns, table, condition = None):
		if condition == None:
			query = f'SELECT {columns} FROM {table}'
		else:
			query = f'SELECT {columns} FROM {table} WHERE {condition}'
			
		try:
			df = pd.read_sql(query, self.connection,index_col='index')
		except :
			df = pd.read_sql(query, self.connection)

		try:
			df.index = [datetime.fromisoformat(i) for i in df.index ]
		finally:
			return df
			
	def get_dataset(self, columns, tables):
		tables_list = tables.split('_')
		tables = tables_list.pop(0)
		for i in tables_list:
			tables += f' INNER JOIN {i} USING("index")'

		query = f'SELECT {columns} FROM {tables}'

		df = pd.read_sql(query, self.connection,index_col='index')
		df.index = [datetime.fromisoformat(i) for i in df.index ]
		df = df.join(self.get_data('*','smp'))

		return df

	def get_metrics(self,model):
		metrics = self.get_data('*',f'metrics_{model}')
		# metrics.index = pd.to_datetime(metrics.index)
		return metrics
		