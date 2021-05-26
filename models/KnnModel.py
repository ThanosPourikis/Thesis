from math import sqrt
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


class KnnModel:
    def __init__(self, validation_size, data):
        self.validation_size = validation_size
        self.data = data

    def knn(self):
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # data = scaler.fit_transform(self.data)
        data = self.data
        del data['Date']
        del data['Unnamed: 0']
        train_size = int(self.validation_size * len(data))

        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=self.validation_size)
        x_train, x_validate, y_train, y_validate = x[:train_size], x[train_size:], y[:train_size], y[train_size:]
        regressor = KNeighborsRegressor(n_neighbors=50)
        regressor.fit(x_train, y_train)
        x_prediction = regressor.predict(x_train)

        print(f'Train Root Mean Square Error : {sqrt(mean_squared_error(y_train,x_prediction))}')

        y_prediction = regressor.predict(x_validate)
        print(f'Validate Root Mean Square Error : {sqrt(mean_squared_error(y_validate, y_prediction))}')


