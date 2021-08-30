import pandas as pd
from utils.database_interface import DB
# dataset = 'requirements_units'

# models = ['Linear','Knn','XgB','Lstm','Hybrid_Lstm']
# train = pd.DataFrame()
# validation = pd.DataFrame()
# test = pd.DataFrame()
# for i in models :
#   df = pd.read_json(f'http://127.0.0.1:5000/metrics_api/{i}')
#   train[i] = df['Train']
#   validation[i] = df['Validation']
#   test[i] = df['Test']
  
# train.to_excel(f'metrics/train_{dataset}.xlsx'),
# validation.to_excel(f'metrics/validation_{dataset}.xlsx'),
# test.to_excel(f'metrics/test_{dataset}.xlsx')

# db = DB()
# df = pd.read_json('http://127.0.0.1:5000/prices_api')
# from sklearn.metrics import mean_absolute_percentage_error
# df['SMP'] = [93.05,90.00,88.52,81.44,88.52,90.00,96.35,108.46,113.00,109.00,107.54,107.54,107.57,107.52,100.54,99.04,107.56,113.00,139.99,128.00,127.00,109.11,125.98,104.54]
# for i in df:
#   print(i,mean_absolute_percentage_error(df['SMP'],df[i]))
# df