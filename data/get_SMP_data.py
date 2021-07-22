from data.get_SMP_files import get_SMP_files
from datetime import timedelta
from os import listdir
import time
import pandas as pd
from pytz import timezone

def get_SMP_data(new_files = True):
	if new_files:
		get_SMP_files()
		
	localTz = timezone('CET')
	folder_path = 'smp_files/'
	files = [f for f in listdir(folder_path)]
	export = pd.DataFrame()
	for name in files:
		try:
			df = pd.read_excel(folder_path + name,header=None)
			line = (df[df.iloc[:,0] == 'Market Clearing Price']).index[0] +1
			temp = pd.DataFrame(df.iloc[line][1:-1]).dropna().reset_index(drop = True)
			
			# if len(temp) == 23:#Daylight saving 02:00:00 is missing so i aproximate the values from the previous and next hours
			# 	temp = temp.append(pd.DataFrame(temp.iloc[1:3].mean(axis=0).to_dict(),index=[2.5])).sort_index().dropna().reset_index(drop = True)
			# elif len(temp) == 25:#Daylight saving 25hours mean 
			# 	temp.iloc[3] = temp.iloc[1:3].mean(axis=0)
			# 	temp = temp.drop([4]).sort_index().reset_index(drop = True)
			if len(temp) == 24:
				temp['Date'] = df.iloc[1][0]
				print(f'Proccessing {name}')
				for i in range(len(temp['Date'])):
					temp.loc[i,'Date'] += timedelta(hours = i)
				temp['Date'] = [localTz.localize(x) for x in temp['Date']]
				
				export= export.append(temp,ignore_index=True)
		except:
			print('Not xlsx File')

	export.columns = ['SMP','Date']
	export = export.set_index('Date').sort_index()
	export.to_csv('datasets/SMP.csv')
	return export.reset_index()
	

# pool = Pool()
# pool.map(read_xlsx, files)

