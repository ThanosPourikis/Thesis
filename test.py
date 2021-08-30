import pandas as pd

dataset = 'requirements_units_weather'

models = ['Linear','Knn','XgB','Lstm','Hybrid_Lstm']
train = pd.DataFrame()
validation = pd.DataFrame()
test = pd.DataFrame()
for i in models :
  df = pd.read_json(f'http://127.0.0.1:5000/metrics_api/{i}')
  train[i] = df['Train']
  validation[i] = df['Validation']
  test[i] = df['Test']
  
train.to_excel(f'metrics/train_{dataset}.xlsx'),
validation.to_excel(f'metrics/validation_{dataset}.xlsx'),
test.to_excel(f'metrics/test_{dataset}.xlsx')