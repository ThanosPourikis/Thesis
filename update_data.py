from utils.database_interface import DB
from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp1_data import get_isp_data
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data, get_weather_data
import config

def update():
	db = DB()

	req = get_isp_data(get_excel_data(folder_path=config.ISP1['folder_path'],filetype=config.ISP1['filetype']))
	units = get_unit_data(get_excel_data(folder_path=config.UNITS['folder_path'],filetype=config.UNITS['filetype']))


	download_weather_data()
	weather =get_weather_data()

	Smp = get_SMP_data()


	db.save_df_to_db(dataframe=req,df_name='requirements')
	db.save_df_to_db(dataframe=units,df_name='units')
	db.save_df_to_db(dataframe=Smp,df_name='smp')
	db.save_df_to_db(dataframe=weather,df_name='weather')

	dataset = (req.set_index('Date').join(Smp.set_index('Date'))).reset_index()
	db.save_df_to_db(dataframe=dataset,df_name='dataset')
	dataset = (req.set_index('Date').join(units.set_index('Date')).join(Smp.set_index('Date')))
	dataset.to_csv('training_set.csv')
	db.save_df_to_db(dataframe=dataset,df_name='train_set')



# update()