import numpy as np
import pandas as pd


# def split_data(data, lookback, test_size=0.2):
#
#     test_size = int(np.round(data.shape[0] * test_size)/24)
#     train_size = int((data.shape[0] - test_size)/24)
#     windows = []
#     for i in range(lookback, train_size, lookback):
#         # windows.append(data_raw[i: i + lookback])
#         windows.append(data[i: i + lookback])
#
#     windows = np.array(windows)
#
#     x_train = windows[:train_size, :, :-1]
#     y_train = windows[:train_size, :, -1]
#
#     x_validate = windows[train_size:, :, :-1]
#     y_validate = windows[train_size:, :, -1]
#     return x_train, y_train, x_validate, y_validate

# input shape = (batch_size, seq_len, features)
def split_data(data, lookback, validation_size=0.2):

    validate_size = int(len(data) * validation_size)
    train_size = len(data) - validate_size
    y_train = np.array([data.iloc[i: i + lookback, -1] for i in range(lookback, train_size, lookback)])
    y_validate = np.array([data.iloc[i: i + lookback, -1] for i in range(train_size, int(len(data)/24)*24, lookback)][:-1])

    x_train = np.array([data.iloc[i: i + lookback] for i in range(lookback, train_size, lookback)])
    x_validate = np.array([data.iloc[i: i + lookback] for i in range(train_size, int(len(data)/24)*24, lookback)][:-1])
    return x_train, y_train, x_validate, y_validate
