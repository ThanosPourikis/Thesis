


def training_pipeline(features, labels):
    random_state = 42
    del features["Date"]
    del labels["Date"]
    X_train = features.values.astype(float)
    y_train = labels.values.astype(float)
    XX, YY = X_train, y_train
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=False)
    print("tuned xgb regression")
    parameters = {
        "learning_rate": [0.09],
        "max_depth": [3, 4, 5, 6],
        "n_estimators": [100, 1000],
        "colsample_bytree": [0.8],
        "random_state": [random_state],
    }
    model = XGBRegressor()
    clf = GridSearchCV(model, param_grid=parameters, verbose=2)
    clf.fit(X_train, y_train)
    print("best estimator train results", mean_absolute_error(clf.predict(X_train), y_train))
    print("best estimator validation results", mean_absolute_error(clf.predict(X_test), y_test))
    print("==========")
    print("Re-fit with full dataset")
    model = XGBRegressor(**clf.best_params_)
    model.fit(XX, YY, verbose=True)
    return model