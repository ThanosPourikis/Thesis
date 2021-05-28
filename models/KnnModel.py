import time

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


class KnnModel:
    def __init__(self, validation_size, features, labels, n_neighbors_parameters):
        self.validation_size = validation_size
        self.features = features
        self.labels = labels
        self.n_neighbors_parameters = {'n_neighbors': range(1, n_neighbors_parameters)}

    def knn(self):

        del self.features['Date']
        del self.labels['Date']

        x_train, x_validate, y_train, y_validate = train_test_split(self.features, self.labels, random_state=96,
                                                                    test_size=self.validation_size, shuffle=False)
        start_time = time.time()
        gs = GridSearchCV(KNeighborsRegressor(), self.n_neighbors_parameters)
        gs.fit(x_train, y_train)
        print(f'Time:{time.time() - start_time}')

        y_train_prediction = gs.predict(x_train)
        print(f'Mean Absolute Train Error : {mean_absolute_error(y_train, y_train_prediction)}')
        print(f'Mean Absolute Validation Error : {mean_absolute_error(y_validate, gs.predict(x_validate))}')

        print(gs.best_estimator_)
