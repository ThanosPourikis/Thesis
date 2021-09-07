import pandas as pd
from utils.database_interface import DB
from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp_data import get_isp_data
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data, get_weather_data,get_weather_mean
import config

def update():
	db = DB('dataset')
	try:
		start_date = db.get_data('MAX("index")','isp1').values[0,0]
		req = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype'],start_date = start_date))
		req = pd.concat([db.get_data('*','isp1'),req])
	except:
		start_date = '2020-11-01'
		req = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype'],start_date = start_date))

	req.to_csv('datasets/requirements.csv')
	db.save_df_to_db(dataframe=req.copy(),df_name='isp1')

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

	Smp = get_SMP_data()
	db.save_df_to_db(dataframe=Smp.copy(),df_name='smp')



	dataset = req.join(Smp)
	db.save_df_to_db(dataframe=dataset,df_name='requirements')
	dataset = req.join(units).join(Smp)
	db.save_df_to_db(dataframe=dataset,df_name='requirements_units')
	dataset = req.join(weather).join(Smp)
	db.save_df_to_db(dataframe=dataset,df_name='requirements_weather')
	dataset = req.join(units).join(weather).join(Smp)
	db.save_df_to_db(dataframe=dataset,df_name='requirements_units_weather')

	db = DB('requirements')
	db.save_df_to_db(dataframe=req.join(Smp)[-7*24:],df_name='requirements')

	db = DB('requirements_units')
	db.save_df_to_db(dataframe=req.join(Smp)[-7*24:],df_name='requirements_units')

	db = DB('requirements_weather')
	db.save_df_to_db(dataframe=req.join(weather).join(Smp)[-7*24:],df_name='requirements_weather')

	db = DB('requirements_units_weather')
	db.save_df_to_db(dataframe=req.join(weather).join(Smp)[-7*24:],df_name='requirements_units_weather')