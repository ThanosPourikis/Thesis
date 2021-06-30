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
			temp = pd.DataFrame(df.iloc[line][1:25])
			temp['Date'] = df.iloc[1][0]
			print(f'Proccessing {name}')
			for i in range(1,len(temp['Date'])+1):
				temp['Date'][i] += timedelta(hours = i-1)
			temp['Date'] = [localTz.localize(x) for x in temp['Date']]
			if len(temp) != 24:
				print('Fuck')
			export= export.append(temp,ignore_index=True)
		except:
			print('Not xlsx File')

	export.columns = ['SMP','Date']
	export.set_index('Date').sort_index().to_csv('datasets/SMP.csv')
	

	

# pool = Pool()
# pool.map(read_xlsx, files)

