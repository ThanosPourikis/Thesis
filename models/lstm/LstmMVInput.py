
import logging
from math import inf
import time
from os import path

import numpy as np
from torch.nn import L1Loss
from torch.optim import Adam
from torch import from_numpy,Tensor

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import copy

from models.utils import get_metrics_df
from models.lstm.Lstm_model import LSTM
from torch.utils.data import DataLoader
from models.lstm.utils import RequirementsSample, sliding_windows
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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
		# data = data.set_index('Date')
		if data.isnull().values.any():
			self.inference = data[-24:]
			self.test = data[-(8*24):-24]
			data = data[:-(8*24)]
		else:
			self.test = data[-(7*24):]
			data = data[:-(7*24)]


		self.features = data.loc[:,data.columns!='SMP']
		self.labels = data['SMP']
		self.input_size = self.features.columns.shape[0]
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


	def train(self):
		self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.features, self.labels, 
		random_state=96,test_size=self.validation_size,shuffle=False)
		
		self.x_train,self.y_train = sliding_windows(self.x_train,self.y_train,sequence_len=self.sequence_length ,window_step=1)
		self.x_validate,self.y_validate = sliding_windows(self.x_validate,self.y_validate,sequence_len=self.sequence_length ,window_step=1)
		
		feature_t_s = MinMaxScaler(feature_range=(-1, 1))
		feature_v_s = MinMaxScaler(feature_range=(-1, 1))
		labels_t_s = MinMaxScaler(feature_range=(-1, 1))
		labels_v_s = MinMaxScaler(feature_range=(-1, 1))

		self.x_train = feature_t_s.fit_transform(self.x_train.reshape(-1, self.x_train.shape[-1])).reshape(self.x_train.shape)
		self.x_validate = feature_v_s.fit_transform(self.x_validate.reshape(-1, self.x_validate.shape[-1])).reshape(self.x_validate.shape)

		self.y_train = labels_t_s.fit_transform(self.y_train.squeeze())
		self.y_validate = labels_v_s.fit_transform(self.y_validate.squeeze())

		model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size,output_dim = self.output_dim,
						num_layers=self.num_layers,batch_first=True)



		self.criterion = L1Loss()
		optimiser = Adam(model.parameters(), self.learning_rate)

		self.error_train = np.empty(0)
		self.error_val = np.empty(0)
		self.y_train_pred_arr = np.empty([0])
		y_val_pred_arr = np.empty([0])
		start_time = time.time()

		train_data_loader = DataLoader(RequirementsSample(self.x_train,self.y_train),self.batch_size, shuffle=False,drop_last=True)
		val_data_loader = DataLoader(RequirementsSample(self.x_validate,self.y_validate),self.batch_size, shuffle=False,drop_last=True)
		while(True): # Early Stopping If error hasnt decrased in 50 epochs STOP
		# for i in range(self.num_epochs):
			err = []
			temp = np.empty(0)
			for j, k in train_data_loader:
				model.train()
				y_train_pred = model(j.float())
				temp = np.append(temp,y_train_pred.detach().numpy().squeeze())
				loss = self.criterion(y_train_pred.squeeze(), k.squeeze().float())
				err.append(loss.detach().item())
				optimiser.zero_grad()
				loss.backward()
				optimiser.step()
			self.y_train_pred_arr = np.append(self.y_train_pred_arr,temp)
			self.error_train = np.append(self.error_train, (sum(err) / len(err)))

			model.eval()
			err = []
			temp = np.empty(0)
			for j, k in val_data_loader:
				y_val_pred = model(j.float())
				temp = np.append(temp,y_val_pred.detach().numpy().squeeze())
				loss = self.criterion(y_val_pred.squeeze(),k.squeeze().float())
				err.append(loss.detach().item())
			y_val_pred_arr = np.append(y_val_pred_arr,temp)		
			self.error_val = np.append(self.error_val,(sum(err)/len(err)))

			if self.error_val[-1] <= self.error_val.min() :
				self.model = copy.deepcopy(model)

			logging.info(f"{self.name}\t Time {time.time() - start_time:.4f}\t Epoch {self.error_val.shape[0]} {self.loss_function} \t Training\t{self.error_train[-1]:.4f}\t Validation\t{self.error_val[-1]:.4f}")
			
			if (self.error_val.shape[0] - self.error_val.argmin()) > self.num_epochs:
				break

		self.best_epoch = self.error_val.argmin()
		logging.info(f'{self.name} Training Completed Best_epoch : {self.best_epoch} Training Time {time.time() - start_time:.4f}')

		self.y_train_pred_best = np.array(self.y_train_pred_arr.reshape(self.error_train.shape[0],-1,32,24)[self.best_epoch]).reshape(-1,24)
		y_val_pred_best = np.array(y_val_pred_arr.reshape(self.error_train.shape[0],-1,32,24)[self.best_epoch]).reshape(-1,24)

		self.y_train_prediction = labels_t_s.inverse_transform(self.y_train_pred_best)
		self.y_train_denorm = labels_t_s.inverse_transform(self.y_train) 

		self.y_val_pred_denorm = labels_v_s.inverse_transform(y_val_pred_best)
		self.y_validate_denorm = labels_v_s.inverse_transform(self.y_validate)

	def get_results(self,test = None):
		if test==None:
			test = self.test
			
		train_error = mean_absolute_error(self.y_train_denorm[:self.y_train_prediction.shape[0]],self.y_train_prediction)
		validate_error = mean_absolute_error(self.y_validate_denorm[:self.y_val_pred_denorm.shape[0]],self.y_val_pred_denorm)
		logging.info(f'{self.name} Best Epoch {self.best_epoch} Train score : {train_error:.4f} Val Score : {validate_error:.4f}')
		hist = pd.DataFrame()
		hist['hist_train'] = self.error_train.tolist()
		hist['hist_val'] = self.error_val.tolist()
		
		
		x_test,y_test = sliding_windows(test.loc[:,test.columns != 'SMP'],test.loc[:,'SMP'],sequence_len=24,window_step=24)
		
		x_test_scaler = MinMaxScaler(feature_range=(-1, 1))
		x_test = x_test_scaler.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
		# x_test = torch.from_numpy(x_test).type(torch.Tensor)
		
		y_test_scaler = MinMaxScaler(feature_range=(-1, 1))
		y_test = y_test_scaler.fit_transform(y_test)
		# y_test = torch.from_numpy(y_test).type(torch.Tensor)
		
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
		y_test = y_test_scaler.inverse_transform(y_test)

		metrics = get_metrics_df(
			self.y_train_denorm[:self.y_train_prediction.shape[0]],self.y_train_prediction,
			self.y_validate_denorm[:self.y_val_pred_denorm.shape[0]],self.y_val_pred_denorm,
			y_test,test_pred)

		self.test['Prediction'] = test_pred.flatten()

		try:
			x_infe,_ = sliding_windows(self.inference.loc[:,self.inference.columns != 'SMP'],self.inference.loc[:,'SMP'],sequence_len=24,window_step=24)
			infe_scaler = MinMaxScaler(feature_range=(-1, 1))
			x_infe = infe_scaler.fit_transform(x_infe.squeeze()).reshape(x_infe.shape)
			x_infe = from_numpy(x_infe).type(Tensor)
			self.model.eval()
			pred_arr = self.model(x_infe)
			self.inference['Inference'] = y_test_scaler.inverse_transform(pred_arr.detach().numpy().reshape(1,-1)).flatten()
			self.test = pd.concat([self.test,self.inference['Inference']],axis=1)
			export = self.test.loc[:,['SMP','Prediction','Inference']]
			return export,metrics,hist,self.best_epoch
		except:
			export = pd.DataFrame()
			export = self.test.loc[:,['SMP','Prediction']]
			return export,metrics,hist,self.best_epoch

		