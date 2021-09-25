import pandas as pd
from utils.database_interface import DB
from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp_data import get_isp_data
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data, get_weather_data,get_weather_mean
import config

def update(new_data):
	db = DB('dataset')
	try:
		start_date = db.get_data('MAX("index")','requirements').values[0,0]
		requirements = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype'],start_date = start_date))
		requirements = pd.concat([db.get_data('*','requirements'),requirements])
	except:
		start_date = '2020-11-01'
		requirements = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype'],start_date = start_date))

	requirements.to_csv('datasets/requirements.csv')
	db.save_df_to_db(dataframe=requirements.copy(),df_name='requirements')

	try:
		start_date = db.get_data('MAX("index")','units').values[0,0]
		units = get_unit_data(get_excel_data(folder_path=config.UNITS['folder_path'],filetype=config.UNITS['filetype'],start_date = start_date))
		if not units.empty:
			units = pd.concat([db.get_data('*','units')[units.columns],units]).fillna(0)
		else:
			units = db.get_data('*','units')
	except:
		start_date = '2020-11-01'
		units = get_unit_data(get_excel_data(folder_path=config.UNITS['folder_path'],filetype=config.UNITS['filetype'],start_date = start_date))

	units.to_csv('datasets/units.csv')
	db.save_df_to_db(dataframe=units.copy(),df_name='units')


	download_weather_data()
	weather =get_weather_mean()
	db.save_df_to_db(dataframe=weather.copy(),df_name='weather')

	Smp = get_SMP_data(new_data)
	db.save_df_to_db(dataframe=Smp.copy(),df_name='smp')



	db = DB('requirements')
	db.save_df_to_db(dataframe=requirements.join(Smp),df_name='requirements')

	db = DB('requirements_units')
	db.save_df_to_db(dataframe=requirements.join(units).join(Smp),df_name='requirements_units')

	db = DB('requirements_weather')
	db.save_df_to_db(dataframe=requirements.join(weather).join(Smp),df_name='requirements_weather')

	db = DB('requirements_units_weather')
	db.save_df_to_db(dataframe=requirements.join(units).join(weather).join(Smp),df_name='requirements_units_weather')