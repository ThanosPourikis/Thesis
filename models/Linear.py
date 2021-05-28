from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split


class Linear:
    def __init__(self, features, labels, validation_size=0.2):
        self.features = features
        self.labels = labels
        self.validation_size = validation_size

    def rum_linear(self):
        # model = LinearRegression()
        del self.features['Date']
        del self.labels['Date']
        x_train, x_validate, y_train, y_validate = train_test_split(self.features, self.labels, random_state=69,
                                                                    test_size=self.validation_size, shuffle=False)
        lr = LinearRegression().fit(x_train, y_train)
        print(f'Train Score :{lr.score(x_train,y_train)}')
        print(f'Validation Score : {lr.score(x_validate,y_validate)}')

