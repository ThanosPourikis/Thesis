
from datetime import datetime

import pandas as pd
import requests
import os

dt = datetime.now()
yyyy = dt.year
mm = dt.month
dd = dt.day
folder_path = 'isp1_results/'

url = f'https://www.admie.gr/getOperationMarketFilewRange?dateStart=2020-11-01&dateEnd={yyyy}-{mm}-{dd}&FileCategory=ISP1ISPResults'

if os.path.exists(folder_path):
    print('Directory already exists')
else:
    print('Creating Directory')
    os.mkdir(folder_path)

data = requests.get(url)
for file in data.json():
    temp = file['file_path'].split('/')[-1]
    if os.path.exists(folder_path+temp):
        print(f'File : {temp} exists')
    else:
        print(f'Downloading File : {temp}')
        with open(folder_path + temp ,'wb') as xlsx:
            xlsx.write(requests.get(file['file_path']).content)

print(file['file_description'])