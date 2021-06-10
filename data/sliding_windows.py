import numpy as np
import pandas as pd


# def split_data(data, lookback, test_size=0.2):

#     test_size = int(int(np.round(data.shape[0]) * test_size))
#     train_size = np.round(data.shape[0]) - test_size
#     windows = []
#     data = np.array(data)
#     final_number = train_size + test_size
#     for i in range(lookback, final_number, lookback):
#         # windows.append(data_raw[i: i + lookback])
#         windows.append(data[i: i + lookback])

#     windows = np.array(windows)
#     next_window = pd.DataFrame(windows.reshape(-1, 24)).shift(-1)
#     next_window = next_window.fillna(0)
#     next_window = next_window.values.reshape(-1, 24, 1)

#     train_size = int(train_size/24)
#     test_size = int(test_size/24)

#     x_train = windows[:train_size, :, -1]
#     y_train = next_window[:train_size, :, -1]

#     x_validate = windows[train_size:, :, -1]
#     y_validate = next_window[train_size:, :, -1]
#     return x_train, y_train, x_validate, y_validate


# input shape = (batch_size, seq_len, features)
def split_data(data, lookback, validation_size=0.2):

    validate_size = int(len(data) * validation_size)
    train_size = len(data) - validate_size
    y_train = np.array([data.iloc[i: i + lookback, -1] for i in range(lookback, train_size+24, lookback)])
    y_validate = np.array([data.iloc[i: i + lookback, -1] for i in range(train_size+24, (int(len(data)/24)*24), lookback)][:-1])

    x_train = np.array([data.iloc[i: i + lookback] for i in range(0, train_size, lookback)])
    x_validate = np.array([data.iloc[i: i + lookback] for i in range(train_size, (int(len(data)/24)*24)-24, lookback)][:-1])
    return x_train, y_train, x_validate, y_validate
