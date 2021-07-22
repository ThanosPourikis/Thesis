
import time
from os import path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy

from models.lstm.Lstm_model import LSTM
from torch.utils.data import DataLoader
from models.lstm.utils import RequirementsSample, slinding_windows, split_data
from sklearn.metrics import mean_absolute_error



class LstmMVInput:
	def __init__(self, loss_function, data,
				 learning_rate=0.001,
				 sequence_length=24,
				 batch_size = 24,
				 hidden_size=128,
				 num_layers=1,
				 output_dim=1,
				 num_epochs=150,
				 model = None):

		self.loss_function = loss_function
		self.data = data
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.model = model


	def train(self):
		self.data = (self.data).loc[:,self.data.columns!='Date']

		self.data = self.data.reset_index(drop=True).dropna()
		self.input_size = len(self.data.columns) - 1
		

		feature_scaler = MinMaxScaler(feature_range=(-1, 1))
		labels_scaler = MinMaxScaler(feature_range=(-1, 1))

		# dd = data.DataLoader(RequirementsLoader(self.data),24)
 
		print('No preTrained Model Found ..... \nTraining .....')

		model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size,output_dim = self.output_dim,
						num_layers=self.num_layers)
		# choose loss function
		criterion = torch.nn.L1Loss()
		optimiser = torch.optim.Adam(model.parameters(), self.learning_rate)


		error_train = np.zeros([self.num_epochs])
		error_val = np.zeros([self.num_epochs])
		
		x_train, y_train, x_validation, y_validation = slinding_windows(self.data[:-24],self.sequence_length)
		# x_train, y_train, x_validation, y_validation = split_data(self.data,self.sequence_length)

		x_train = feature_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
		x_validation = feature_scaler.transform(x_validation.reshape(-1, x_validation.shape[-1])).reshape(x_validation.shape)

		y_train = labels_scaler.fit_transform(y_train.squeeze())
		y_validation = labels_scaler.transform(y_validation.squeeze())

		train_data_loader = DataLoader(RequirementsSample(x_train,y_train),self.batch_size,drop_last=True)
		val_data_loader = DataLoader(RequirementsSample(x_validation,y_validation),self.batch_size,drop_last=True)
		
		y_train_pred_arr = list()
		y_val_pred_arr = list()
		models_dict = dict()
		
		for i in range(self.num_epochs):
			err = []

			for j, k in train_data_loader:
				model.train()
				temp = list()
				y_train_pred = model(j.float())
				temp.append(y_train_pred.detach().numpy().squeeze())
				loss = criterion(y_train_pred.squeeze(), k.squeeze().float())
				err.append(loss.detach().item())
				optimiser.zero_grad()
				loss.backward()
				optimiser.step()
			y_train_pred_arr.append(temp)
			error_train[i] = sum(err) / len(err)

			err = []
			with torch.set_grad_enabled(False):
				for j, k in val_data_loader:
						temp = list()
						model.eval()
						y_val_pred = model(j.float())
						temp.append(y_val_pred.detach().numpy().squeeze())
						loss = criterion(y_val_pred.squeeze(),k.squeeze())
						err.append(loss.detach().item())
				y_val_pred_arr.append(temp)
			error_val[i] = sum(err)/len(err)
			models_dict[i] = copy.deepcopy(model)
			print(f"Epoch\t Training\t {i} {self.loss_function}  {error_train[i]}\t Validation\t {i} {self.loss_function}  {error_val[i]}")

		# invert predictions
		y_train_pred = labels_scaler.inverse_transform(y_train_pred.detach().numpy().squeeze())
		y_train = labels_scaler.inverse_transform(y_train)
		y_val_pred = labels_scaler.inverse_transform(y_val_pred.detach().numpy().squeeze())
		y_validation = labels_scaler.inverse_transform(y_validation)

		self.model = models_dict[error_val.argmin()]


		hist = pd.DataFrame()
		hist['train'] = error_train
		hist['val'] = error_val
		return y_val_pred.squeeze(),hist,models_dict[hist['val'].argmin()]
		