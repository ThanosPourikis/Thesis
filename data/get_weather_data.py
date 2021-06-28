from pandas._libs.tslibs.timestamps import Timestamp
from  config import Cities,dark_sky_key
import requests
import json
from pytz import timezone
from datetime import datetime,timedelta,date
import pandas as pd

def download_weather_data():

	exclude = 'exclude=currently,minutely,daily,alerts,flags'
	to_keep = {'time' : 0,'temperature' : 0,'windSpeed' : 0,'windGust' : 0,'cloudCover' : 0,'uvIndex' : 0,'visibility' : 0,'city' : ''}
	feature_list = ['time','temperature','windSpeed','windGust','cloudCover','uvIndex','visibility']

	end = datetime.date(datetime.now() + timedelta(days=1))
	localTz = timezone('CET')
	df = pd.DataFrame(columns=to_keep)
	time = int(datetime.fromisoformat('2020-11-01 00:00:00').timestamp())

	timedel = 86400

	for city in Cities:
		try:
			df = pd.read_csv(f'datasets/weather_data_{city}.csv')
			time = int(datetime.fromisoformat(df.iloc[-1]['time']).timestamp())
		except :
			df = pd.DataFrame(columns=to_keep)
			time = int(datetime.fromisoformat('2020-11-01 00:00:00').timestamp())

		
		lat = Cities[city][0]
		lon = Cities[city][1]
		to_keep['city'] = city
		while date.fromtimestamp(time) != end:
			url = f'https://api.darksky.net/forecast/{dark_sky_key}/{lat},{lon},{time}?units=si&{exclude}'
			response = requests.get(url)
			k = json.loads(response.text)
			for i in k['hourly']['data']:
				for j in feature_list:
					to_keep[j] = i[j]
				to_keep['time'] = localTz.localize(datetime.fromtimestamp(to_keep['time']))
				df = df.append(to_keep,ignore_index=True)
			time += timedel
		df.set_index('time').to_csv(f'datasets/weather_data_{city}.csv')
	

def get_weather_data():
	export = pd.DataFrame(columns=['temperature','windSpeed','windGust','cloudCover','uvIndex','visibility'],index=['time'])

	for city in Cities:
		df = pd.read_csv(f'datasets/weather_data_{city}.csv',index_col=0)
		df = df.loc[:,df.columns!='city']
		return df
