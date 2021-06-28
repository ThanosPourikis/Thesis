
from datetime import datetime,timedelta

import pandas as pd
from pandas.core.frame import DataFrame
import requests
import os
from pytz import timezone

localTz = timezone('CET')
dt = datetime.now() + timedelta(days=1)


def get_excel_data(folder_path = 'all_files_isp1/',filetype = 'ISP1Requirements'):
	df = {}

	yyyy = dt.year
	mm = dt.month
	dd = dt.day
	

	if os.path.exists(folder_path):
		print('Directory already exists')
	else:
		print('Creating Directory')
		os.mkdir(folder_path)

	url = f'https://www.admie.gr/getOperationMarketFilewRange?dateStart=2020-11-01&dateEnd={yyyy}-{mm}-{dd}&FileCategory={filetype}'

	name = ''
	data = requests.get(url)
	for file in data.json():
		if name[:8] != file['file_path'].split('/')[-1][:8] :
			name = file['file_path'].split('/')[-1][:8] + '.xlsx'
			if os.path.exists(folder_path+name):
				print(f'Reading File : {name}')
				df[name] = pd.read_excel(folder_path+name)
			else:
				print(f'Downloading File : {name}')
				with open(folder_path + name,'wb') as xlsx:
					xlsx.write(requests.get(file['file_path']).content)
				df[name] = pd.read_excel(folder_path+name)
	return df

def get_isp_data(df):
	export_df = pd.DataFrame()
	for i in df:
		temp = pd.DataFrame()
		
		date = datetime.fromisoformat(f'{i[0:4]}-{i[4:6]}-{i[6:8]}')
		date = [date + timedelta(hours = i) for i in range(24)]
		date = [localTz.localize(x) for x in date]
		del date[3]

		res = df[i].iloc[df[i][df[i].iloc[:,0] == 'Non-Dispatchable RES'].index[0] + 1][2:-1]
		temp['Renewables'] = res

		nndis = df[i].iloc[df[i][df[i].iloc[:,0] == 'Non-Dispatcheble Losses'].index[0] + 1]
		temp['Non-Dispatcheble'] = nndis

		man_hydro = df[i].iloc[df[i][df[i].iloc[:,0] == 'Commissioning'].index[0] - 1]
		temp['Man_Hydro'] = man_hydro

		comm = df[i].iloc[df[i][df[i].iloc[:,0] == 'Commissioning'].index[0] + 1]
		temp['Commissioning'] = comm

		req_start = df[i][df[i].iloc[:,0] == 'Up'].index[0]
		req_end = req_start + 3

		temp['Reserve_Up'] = df[i].iloc[req_start:req_end,:-1].sum()[2:-1]

		req_start = df[i][df[i].iloc[:,0] == 'Down'].index[0]
		req_end = req_start + 3

		temp['Reserve_Down'] = df[i].iloc[req_start:req_end,:-1].sum()[2:-1]


		pairs=[]
		for j in range(0,len(temp),2):
			pairs.append(temp.iloc[j:j+2].mean(axis=0)) #Example (00:00:00 + 00:30:00)/2 -> 00:00:00
		pairs = pd.DataFrame(pairs)
		
		if len(pairs) == 24:
			pairs=pairs.drop([3]).reset_index(drop=True)
		
		pairs["Date"] = date

		
		
		pairs = pairs.set_index('Date')
		export_df = export_df.append(pairs)
	export_df.sort_index().to_csv('datasets/requirements.csv')
