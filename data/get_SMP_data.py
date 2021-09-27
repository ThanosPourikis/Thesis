from data.get_SMP_files import get_SMP_files
from datetime import timedelta,datetime
from os import listdir
import pandas as pd
from pytz import timezone

localTz = timezone('CET')
dt = localTz.localize(datetime.now())  + timedelta(days = 1)
folder_path = 'smp_files/'

def get_SMP_data(start_date = None):
	get_SMP_files()
	files = []
	if  start_date == None:
		files = [f for f in listdir(folder_path)]
	else:
		start_date = datetime.fromisoformat(start_date)
		days = (dt - start_date).days
		for i in range(days):
			date = start_date + timedelta(days=i)
			files.append(f"{str(date)[:10].replace('-','')}_EL-DAM_ResultsSummary_EN_v01.xlsx")

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
					temp.loc[i,'Date'] += timedelta(hours = i+1)
				temp['Date'] = [localTz.localize(x) for x in temp['Date']]
				temp = temp.set_index('Date')
				export = pd.concat([export,temp])
		except:
			print('Not xlsx File')
	try:
		export.columns = ['SMP']
		export = pd.to_numeric(export['SMP'])
		export = export.sort_index()
		return pd.DataFrame(export)
	except:
		 return pd.DataFrame()

# pool = Pool()
# pool.map(read_xlsx, files)

