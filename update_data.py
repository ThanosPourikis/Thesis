from data.get_SMP_data import get_SMP_data
from data.units_data import get_unit_data
from data.isp1_data import get_isp_data
from data.get_SMP_files import get_SMP_files
from data.ADMHE_files import get_excel_data
from data.get_weather_data import download_weather_data
import config


get_unit_data(get_excel_data(folder_path=config.units['folder_path'],filetype=config.units['filetype']))
download_weather_data()
get_isp_data(get_excel_data(folder_path=config.isp1['folder_path'],filetype=config.isp1['filetype']))
get_SMP_data()

