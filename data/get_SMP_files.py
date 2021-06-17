import os
import  requests
import re
from zipfile import ZipFile

export = []
folder_path = 'smp_files/'


if os.path.exists(folder_path):
	print('Directory already exists')
else:
	print('Creating Directory')
	os.mkdir(folder_path)
#zip file of previous year from 2020-11-01
url = 'https://www.enexgroup.gr/el/c/document_library/get_file?uuid=23d04ca0-5850-415b-4c7b-d0a65ed293cf&groupId=20126'

zip_path = folder_path + '2020_EL-DAM_ResultsSummary.zip'

with open(zip_path,'wb') as zip_file:
	zip_file.write(requests.get(url).content)

with ZipFile(zip_path,'r') as zip_file:
	zip_file.extractall(folder_path)

os.remove(zip_path)


for i in range(53):
	url=f'https://www.enexgroup.gr/el/web/guest/markets-publications-el-day-ahead-market?p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2_delta=7&p_r_p_resetCur=false&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2_cur={i}'

	data = requests.get(url)
	print(data)
	x = re.findall('Αποτελέσματα Aγοράς Επόμενης Ημέρας(.* ?)', data.text)
	names = re.findall('[0-9]{8}_EL-DAM_ResultsSummary_EN_v01\.xlsx',''.join(x))
	urls = re.findall('uuid(.*?)"',''.join(x))
	for i,j in zip(names,urls[:len(names)]):
		url = f'https://www.enexgroup.gr/el/c/document_library/get_file?uuid{j}'
		x = requests.get(url).content
		print(f'Downloading File ')
		with open(folder_path + i ,'wb') as xlsx:
			xlsx.write(x)


# https://www.enexgroup.gr/el/c/document_library/get_file?uuid=08c1813a-100c-42cb-0124-d1f6a7eb3325&groupId=20126