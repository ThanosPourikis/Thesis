import numpy as np


def split_data(data, lookback, test_size=0.2):
    # data_raw = data.to_numpy()

    windows = []
    for i in range(len(data) - lookback):
        # windows.append(data_raw[i: i + lookback])
        windows.append(data[i: i + lookback])

    windows = np.array(windows)
    test_size = int(np.round(data.shape[0]*test_size))
    train_size = data.shape[0] - test_size
    x_train = windows[:train_size, :-1, :]
    y_train = windows[:train_size, -1, :]

    x_test = windows[train_size:, :-1, :]
    y_test = windows[train_size:, -1, :]
    return [x_train, y_train, x_test, y_test]
