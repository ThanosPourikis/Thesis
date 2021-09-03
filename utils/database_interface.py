import sqlalchemy as sq
import pandas as pd
from pytz import timezone
from datetime import datetime,timedelta
import config
localTz = timezone('CET')
sqlite = 'sqlite:///{}.db'
mysql = f'mysql+mysqldb://{config.USERNAME}:{config.PASSWORD}@{config.HOSTNAME}/{config.DATABASENAME}'
class DB:
	def __init__(self,database):
		self.engine = sq.create_engine(sqlite.format(database))
		self.connection = self.engine.connect()
	def save_df_to_db(self, dataframe, df_name):
		try:
			dataframe.index = [str(i) for i in dataframe.index]
		except:
			pass
		dataframe.to_sql(df_name, self.connection, if_exists='replace')

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
			df['SMP'] = pd.to_numeric(df['SMP'])
		except:
			pass
		try:
			df.index = [datetime.fromisoformat(i) for i in df.index ]
			df.index = pd.to_datetime(df.index)
		except:
			pass

		return df
	def save_inference_to_DB():
		pass
	