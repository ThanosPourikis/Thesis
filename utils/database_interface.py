import sqlalchemy
import pandas as pd

class DB:
	def __init__(self):
		self.engine = sqlalchemy.create_engine('sqlite:///database.db')
		self.connection = self.engine.connect()

	def save_df_to_db(self, dataframe, df_name):
		dataframe.to_sql(df_name, self.connection, if_exists='replace')


	def get_data(self, table, columns):
		try:
			df = pd.read_sql(f'SELECT {columns} FROM {table}', self.connection,index_col='index')
		except :
			df = pd.read_sql(f'SELECT {columns} FROM {table}', self.connection)
		df['SMP'] = pd.to_numeric(df['SMP'])
		df['Date'] = pd.to_datetime(df['Date'])
		return df