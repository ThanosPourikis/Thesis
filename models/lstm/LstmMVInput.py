
import logging
import time
from os import path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy

from models.lstm.Lstm_model import LSTM
from torch.utils.data import DataLoader
from models.lstm.utils import RequirementsSample, sliding_windows
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.database_interface import DB

### TODO Add Hybrid Code

class LstmMVInput:
	def __init__(self, loss_function, data,name,
				 learning_rate=0.001,
				 validation_size= 0.2,
				 sequence_length=24,
				 batch_size = 32,
				 hidden_size=128,
				 num_layers=1,
				 output_dim=1,
				 num_epochs=150,
				 model = None):
		data = data.set_index('Date')
		if data.isnull().values.any():
			self.inference = data[-24:]
			self.test = data[-(8*24):-24]
			data = data[:-(8*24)]
		else:
			self.test = data[-(7*24):]
			data = data[:-(7*24)]


		self.features = data.loc[:,data.columns!='SMP']
		self.labels = data.loc[:,data.columns=='SMP']
		self.input_size = len(self.features.columns)
		self.validation_size = validation_size
		self.loss_function = loss_function
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.name = name
		self.model = model
		self.db = DB()


	def train(self):
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels, 
		random_state=96,test_size=self.validation_size,shuffle=False)
		
		x_train,y_train = sliding_windows(self.x_train,self.y_train)
		x_validate,y_validate = sliding_windows(self.x_validate,self.y_validate)
		scalers = {
			'feature_t' : MinMaxScaler(feature_range=(-1, 1)),
			'feature_v' : MinMaxScaler(feature_range=(-1, 1)),
			'labels_t' : MinMaxScaler(feature_range=(-1, 1)),
			'labels_v' : MinMaxScaler(feature_range=(-1, 1))
		}
		x_train = scalers['feature_t'].fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
		x_validate = scalers['feature_v'].fit_transform(x_validate.reshape(-1, x_validate.shape[-1])).reshape(x_validate.shape)

		y_train = scalers['labels_t'].fit_transform(y_train.squeeze())
		y_validate = scalers['labels_v'].fit_transform(y_validate.squeeze())

		model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size,output_dim = self.output_dim,
						num_layers=self.num_layers,batch_first=True)



		self.criterion = torch.nn.L1Loss()
		optimiser = torch.optim.Adam(model.parameters(), self.learning_rate)

		self.error_train = np.zeros([self.num_epochs])
		self.error_val = np.zeros([self.num_epochs])
		y_train_pred_arr = list()
		y_val_pred_arr = list()
		models_dict = dict()
		start_time = time.time()

		train_data_loader = DataLoader(RequirementsSample(x_train,y_train),self.batch_size, shuffle=False)
		val_data_loader = DataLoader(RequirementsSample(x_validate,y_validate),self.batch_size, shuffle=False)
		for i in range(self.num_epochs):
			err = []
			for j, k in train_data_loader:
				model.train()
				temp = list()
				y_train_pred = model(j.float())
				temp.append(y_train_pred.detach().numpy().squeeze())
				loss = self.criterion(y_train_pred.squeeze(), k.squeeze().float())
				err.append(loss.detach().item())
				optimiser.zero_grad()
				loss.backward()
				optimiser.step()
			y_train_pred_arr.append(temp)
			self.error_train[i] = sum(err) / len(err)

			with torch.set_grad_enabled(False):
				model.eval()
				err = []
				for j, k in val_data_loader:
						temp = list()
						y_val_pred = model(j.float())
						temp.append(y_val_pred.detach().numpy().squeeze())
						loss = self.criterion(y_val_pred.squeeze(),k.squeeze().float())
						err.append(loss.detach().item())
				y_val_pred_arr.append(temp)
			self.error_val[i] = sum(err)/len(err)
			models_dict[i] = copy.deepcopy(model)
			logging.info(f"{self.name} Epoch\t Training\t {i} {self.loss_function}  {self.error_train[i]}\t Validation\t {i} {self.loss_function}  {self.error_val[i]}")

		self.best_epoch = self.error_val.argmin()
		logging.info(f'{self.name} Training Completed Best_epoch : {self.best_epoch} Training Time {time.time() - start_time}')
		self.model = models_dict[self.best_epoch]
		y_train_pred_best = y_train_pred_arr[self.best_epoch][0]
		y_val_pred_best = y_val_pred_arr[self.best_epoch][0]

		self.y_train_prediction = scalers['labels_t'].inverse_transform(y_train_pred_best)
		self.y_train_denorm = scalers['labels_t'].inverse_transform(y_train) 

		self.y_val_pred_denorm = scalers['labels_v'].inverse_transform(y_val_pred_best)
		self.y_validate_denorm = scalers['labels_v'].inverse_transform(y_validate)

	def get_results(self):


		train_error = mean_absolute_error(self.y_train_denorm[-len(self.y_train_prediction):],self.y_train_prediction)
		validate_error = mean_absolute_error(self.y_validate_denorm[-len(self.y_val_pred_denorm):],self.y_val_pred_denorm)

		logging.info(f'{self.name} Best Epoch {self.best_epoch} Train score : {train_error} Val Score : {validate_error}')
		hist = pd.DataFrame()
		hist['hist_train'] = self.error_train.tolist()
		hist['hist_val'] = self.error_val.tolist()
		
		
		x_test,y_test = sliding_windows(self.test.loc[:,self.test.columns != 'SMP'],self.test.loc[:,'SMP'],sequence_len=24,window_step=24)
		
		x_test_scaler = MinMaxScaler(feature_range=(-1, 1))
		x_test = x_test_scaler.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
		# x_test = torch.from_numpy(x_test).type(torch.Tensor)
		
		y_test_scaler = MinMaxScaler(feature_range=(-1, 1))
		y_test_scaler.fit_transform(y_test)
		# y_test = torch.from_numpy(y_test).type(torch.Tensor)
		
		with torch.set_grad_enabled(False):
			self.model.eval()
			err = []
			test_data_loader = DataLoader(RequirementsSample(x_test,y_test),1, shuffle=False)
			pred_arr = []
			temp = list()
			for j, k in test_data_loader:
					pred_arr = self.model(j.float())
					temp.append(pred_arr.detach().numpy().squeeze())
					loss = self.criterion(pred_arr.squeeze(),k.squeeze().float())
					err.append(loss.detach().item())
		test_pred = y_test_scaler.inverse_transform(temp)
		test_error = mean_absolute_error(y_test,test_pred)
		self.test['Prediction'] = test_pred.flatten()

		try:
			x_infe,_ = sliding_windows(self.inference.loc[:,self.inference.columns != 'SMP'],self.inference.loc[:,'SMP'],sequence_len=24,window_step=24)
			infe_scaler = MinMaxScaler(feature_range=(-1, 1))
			x_infe = infe_scaler.fit_transform(x_infe.squeeze()).reshape(x_infe.shape)
			x_infe = torch.from_numpy(x_infe).type(torch.Tensor)
			self.model.eval()
			pred_arr = self.model(x_infe)
			self.inference['Inference'] = y_test_scaler.inverse_transform(pred_arr.detach().numpy().reshape(1,-1)).flatten()
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			export = self.test.loc[:,['SMP','Prediction','Inference']]
			return export.reset_index(),train_error,validate_error,test_error,hist
		except:
			export = pd.DataFrame()
			export = self.test.loc[:,['SMP','Prediction']]
			return export.reset_index(),train_error,validate_error,test_error,hist

		