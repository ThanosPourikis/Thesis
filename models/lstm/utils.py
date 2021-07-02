import numpy as np
import pandas as pd

import torch
from data.training_data import get_data_from_csv


# input shape = (batch_size, seq_len, features)

def slinding_windows(data, lookForward, test_size=0.2):

	features = data.loc[:,data.columns != 'SMP']
	labels =  data.loc[:,data.columns == 'SMP']
	xX = []
	yY = []
	for i in range(len(data) - lookForward):

		xX.append(features[i: i + lookForward])
		yY.append(labels[i: i + lookForward])

	

	xX = np.array(xX)
	yY = np.array(yY)

	test_size = int(np.round(data.shape[0]*test_size))
	train_size = data.shape[0] - test_size
	x_train = xX[:train_size]
	y_train = yY[:train_size]

	x_validate = xX[train_size:]
	y_validate = yY[train_size:]
	return x_train, y_train, x_validate, y_validate


def split_data(data, lookForward, validation_size=0.2):

	validate_size = int(len(data) * validation_size)
	train_size = len(data) - validate_size
	y_train = np.array([data.iloc[i: i + lookForward,data.columns =='SMP'] for i in range(lookForward, train_size+24, lookForward)]).squeeze()
	y_validate = np.array([data.iloc[i: i + lookForward,data.columns =='SMP'] for i in range(train_size+24, (int(len(data)/24)*24), lookForward)][:-1]).squeeze()

	x_train = np.array([data.iloc[i: i + lookForward,data.columns !='SMP'] for i in range(0, train_size, lookForward)])
	x_validate = np.array([data.iloc[i: i + lookForward,data.columns !='SMP'] for i in range(train_size, (int(len(data)/24)*24)-24, lookForward)][:-1])
	return x_train, y_train, x_validate, y_validate



class RequirementsSample(torch.utils.data.Dataset):
	def __init__(self,features,labels) -> None:
		self.features = features
		self.labels = labels


	def __len__(self):
		return len(self.features)

	def __getitem__(self,index):
		return torch.tensor(np.array(self.features[index])).double(), torch.tensor(np.array(self.labels[index])).double()


def get_tensors(data, lookforward,x_scaler,y_scaler):
	x_train, y_train, x_validation, y_validation = split_data(data, lookforward)
	# x_train, y_train, x_validation, y_validation = slinding_windows(data, lookforward)

	x_train = x_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
	x_validation = x_scaler.transform(x_validation.reshape(-1, x_validation.shape[-1])).reshape(x_validation.shape)

	y_train = y_scaler.fit_transform(y_train.reshape(-1,y_train.shape[1]))
	y_validation = y_scaler.fit_transform(y_validation.reshape(-1,y_validation.shape[1]))

	x_train = torch.from_numpy(x_train).type(torch.Tensor)
	x_validation = torch.from_numpy(x_validation).type(torch.Tensor)

	y_train = torch.from_numpy(y_train).type(torch.Tensor)
	y_validation = torch.from_numpy(y_validation).type(torch.Tensor)

	return x_train, x_validation, y_train, y_validation

