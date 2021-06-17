
from datetime import datetime,timedelta

import pandas as pd
from pandas.core.frame import DataFrame
import requests
import os
from pytz import timezone

localTz = timezone('CET')
dt = datetime.now() + timedelta(days=1)


def get_excel_data():
	df = {}

	yyyy = dt.year
	mm = dt.month
	dd = dt.day
	folder_path = 'isp1_results/'

	if os.path.exists(folder_path):
		print('Directory already exists')
	else:
		print('Creating Directory')
		os.mkdir(folder_path)

	url = f'https://www.admie.gr/getOperationMarketFilewRange?dateStart=2020-11-01&dateEnd={yyyy}-{mm}-{dd}&FileCategory=ISP1ISPResults'


	data = requests.get(url)

	for file in data.json():
		name = file['file_path'].split('/')[-1]
		if os.path.exists(folder_path+name):
			print(f'Reading File : {name}')
			df[name] = pd.read_excel(folder_path+name)
		else:
			print(f'Downloading File : {name}')
			with open(folder_path + name ,'wb') as xlsx:
				xlsx.write(requests.get(file['file_path']).content)
			df[name] = pd.read_excel(folder_path+name)
	return df



def get_power_generation():

	export_df = pd.DataFrame()
	power_plants = {'AG_DIMITRIOS1' : 'DEH', 'AG_DIMITRIOS2': 'DEH', 'AG_DIMITRIOS3': 'DEH', 'AG_DIMITRIOS4': 'DEH', 'AG_DIMITRIOS5': 'DEH', 'KARDIA3': 'DEH', 
	'KARDIA4' : 'DEH', 'MEGALOPOLI3' : 'DEH', 'MEGALOPOLI4' : 'DEH', 'AMYNDEO1' : 'DEH', 'AMYNDEO2' : 'DEH', 'MELITI' : 'DEH', 'ALIVERI5' : 'DEH', 
	'LAVRIO4' : 'DEH', 'LAVRIO5' : 'DEH', 'KOMOTINI' : 'DEH', 'MEGALOPOLI_V' : 'DEH', 
	'HERON1' : 'HERON', 'HERON2'  : 'HERON' , 'HERON3'  : 'HERON' , 'HERON_CC'  : 'HERON',
	'ELPEDISON_THESS' : 'ELPEDISON', 'ELPEDISON_THISVI' : 'ELPEDISON',
	'ALOUMINIO' : 'MYTILINEOS', 'PROTERGIA_CC' : 'MYTILINEOS', 'KORINTHOS_POWER' : 'MYTILINEOS'}
		
	df = get_excel_data()
	
	for i in df :

		start_df = df[i][df[i].iloc[:,0] == 'AG_DIMITRIOS1'].index[0] -1
		end_df = df[i][df[i].iloc[:,0] == 'Total Thermal Production'].index[0]

		temp = df[i].iloc[start_df:end_df,:-1]
		export = pd.DataFrame()
		export['Date'] = temp.iloc[0,1:].values

		for i in range(1,len(temp)):
			export[temp.iloc[i,0]] = temp.iloc[i,1:].values

		pairs = []

		for i in range(0,len(export),2):
			pairs.append(export.iloc[i:i+2,1:].mean(axis=0)) # Example (00:00:00 + 00:30:00)/2 -> 00:00:00
		pairs = pd.DataFrame(pairs)
		dc = {'DEH' : 0, 'HERON' : 0 ,'ELPEDISON': 0, 'MYTILINEOS' : 0}
		for i in range(len(pairs.columns)):
			dc[power_plants[pairs.iloc[:,i].name]] += pairs.iloc[:,i]# Dictonary[NameOfPlant]  -> Comapany += Produdaction
		dc = pd.DataFrame(dc)
		tempDate = [temp.iloc[0,i] for i in range(1,len(temp.columns), 2)]
		dc['Date'] = [localTz.localize(x) for x in tempDate]
		export_df = export_df.append(dc,ignore_index=True)

	export_df.set_index('Date').sort_index().to_csv('power_generation.csv')


def get_hydro_data():

	df = get_excel_data()

	for i in  df:
		start_df = df[i][df[i].iloc[:,0] == 'System Overview'].index[0]
		end_df = df[i][df[i].iloc[:,0] == 'Mandatory Hydro Injections'].index[0]+1

		temp = df[i].iloc[start_df:end_df,:-1]
		export = pd.DataFrame()
		export['Date'] = temp.iloc[0,1:].values

		for i in range(1,len(temp)):
			export[temp.iloc[i,0]] = temp.iloc[i,1:].values

		pairs=[]
		for i in range(0,len(export),2):
			pairs.append(export.iloc[i:i+2,1:].mean(axis=0)) #Example (00:00:00 + 00:30:00)/2 -> 00:00:00
		pairs = pd.DataFrame(pairs)
		tempDate = [temp.iloc[0,i] for i in range(1,len(temp.columns), 2)]
		pairs["Date"] = [localTz.localize(x) for x in tempDate]
		export_df = export_df.append(pairs,ignore_index=True)

	export_df.set_index('Date').sort_index().to_csv('mandatory_hydro.csv')
