import numpy as np
import pandas as pd


# def split_data(data, lookback, scaler, test_size=0.2):

# 	windows = []
# 	for i in range(len(data) - lookback):
# 		# windows.append(data_raw[i: i + lookback])
# 		windows.append(data[i: i + lookback])

# 	windows = np.array(windows)

# 	test_size = int(np.round(data.shape[0]*test_size))
# 	train_size = data.shape[0] - test_size
# 	x_train = windows[:train_size, :, :]
# 	y_train = windows[:train_size, :, -1]

# 	x_validate = windows[train_size:, :-1, :]
# 	y_validate = windows[train_size:, :, -1]
# 	return x_train, y_train, x_validate, y_validate



# input shape = (batch_size, seq_len, features)
def split_data(data, lookForward =24, validation_size=0.2):

	validate_size = int(len(data) * validation_size)
	train_size = len(data) - validate_size
	y_train = np.array([data.iloc[i: i + lookForward,data.columns =='SMP'] for i in range(lookForward, train_size+24, lookForward)]).squeeze()
	y_validate = np.array([data.iloc[i: i + lookForward,data.columns =='SMP'] for i in range(train_size+24, (int(len(data)/24)*24), lookForward)][:-1]).squeeze()

	x_train = np.array([data.iloc[i: i + lookForward] for i in range(0, train_size, lookForward)])
	x_validate = np.array([data.iloc[i: i + lookForward] for i in range(train_size, (int(len(data)/24)*24)-24, lookForward)][:-1])
	return x_train, y_train, x_validate, y_validate
