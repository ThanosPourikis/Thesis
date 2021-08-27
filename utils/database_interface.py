import sqlalchemy as sq
import pandas as pd
from pytz import timezone
from datetime import datetime,timedelta

localTz = timezone('CET')

class DB:
	def __init__(self):
		engine = sq.create_engine('sqlite:///database.db')
		self.connection = engine.connect()
	def save_df_to_db(self, dataframe, df_name):
		try:
			dataframe.index = [str(i) for i in dataframe.index]
		except:
			pass
		dataframe.to_sql(df_name, self.connection, if_exists='replace')
	def get_data(self, columns, table):
		try:
			df = pd.read_sql(f'SELECT {columns} FROM {table}', self.connection,index_col='index')
		except :
			df = pd.read_sql(f'SELECT {columns} FROM {table}', self.connection)
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
	