from utils.database_interface import DB
from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp_data import get_isp_data
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data, get_weather_data,get_weather_mean
import config

def update():
	db = DB()

	req = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype']))
	db.save_df_to_db(dataframe=req.copy(),df_name='isp1')


	units = get_unit_data(get_excel_data(folder_path=config.UNITS['folder_path'],filetype=config.UNITS['filetype']))
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
