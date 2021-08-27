import pandas as pd


models = ['Linear','Knn','XgB','Lstm','Hybrid_Lstm']
train = pd.DataFrame()
validation = pd.DataFrame()
test = pd.DataFrame()
for i in models :
  df = pd.read_json(f'http://127.0.0.1:5000/metrics_api/{i}')
  train[i] = df['Train']
  validation[i] = df['Validation']
  test[i] = df['Test']
  
train.to_excel('train_requirements_units_weather.xlsx'),
validation.to_excel('validation_requirements_units_weather.xlsx'),
test.to_excel('test_requirements_units_weather.xlsx')