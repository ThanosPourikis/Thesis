import pandas as pd
from utils.database_interface import DB
from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp_data import get_isp_data
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data, get_weather_data,get_weather_mean
import config

def update():
	db = DB()
	start_date = db.get_data('MAX("index")','isp1').values[0,0]
	req = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype'],start_date = start_date))
	req = pd.concat([db.get_data('*','isp1'),req])
	req.to_csv('datasets/requirements.csv')
	db.save_df_to_db(dataframe=req.copy(),df_name='isp1')

	start_date = db.get_data('MAX("index")','units').values[0,0]
	units = get_unit_data(get_excel_data(folder_path=config.UNITS['folder_path'],filetype=config.UNITS['filetype'],start_date = start_date))

	units = pd.concat([db.get_data('*','units'),units]).fillna(0)

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
	dataset = req.join(units).join(weather).join(Smp)
	db.save_df_to_db(dataframe=dataset,df_name='requirements_units_weather')
