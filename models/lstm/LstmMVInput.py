
import time
from os import path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


from models.lstm.Lstm_model import LSTM
from torch.utils import  data
from models.lstm.utils import get_tensors
from utils.utils import loss_function_selection, error_calculation, save_df_to_db


class LstmMVInput:
	def __init__(self, loss_function, data,
				 learning_rate=0.01,
				 lookforward=24,
				 hidden_size=32,
				 num_layers=2,
				 output_dim=24,
				 num_epochs=150,
				 bidirectional = False,
				 dropout = 0.1):

		self.loss_function = loss_function
		self.data = data
		self.lookforward = lookforward
		self.bidirectional = bidirectional
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.dropout = dropout

	def run(self):
		self.data = (self.data).loc[:,self.data.columns!='Date']
		test = (self.data).loc[:,self.data.columns!='SMP'][:-24]

		self.data = self.data.reset_index(drop=True).dropna()
		self.input_size = len(self.data.columns) - 1
		

		feature_scaler = MinMaxScaler(feature_range=(-1, 1))
		labels_scaler = MinMaxScaler(feature_range=(-1, 1))

		# dd = data.DataLoader(RequirementsLoader(self.data),24)
		x_train, x_validation, y_train, y_validation = get_tensors(self.data,self.lookforward,feature_scaler,labels_scaler)
 
		print('No preTrained Model Found ..... \nTraining .....')

		model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size,output_dim = self.output_dim,
						num_layers=self.num_layers,bidirectional = self.bidirectional,dropout = self.dropout)

		# choose loss function
		criterion = loss_function_selection(self.loss_function)
		optimiser = torch.optim.Adam(model.parameters(), self.learning_rate)

		hist_train = np.zeros(self.num_epochs)
		hist_val = np.zeros(self.num_epochs)

		start_time = time.time()

		for t in range(self.num_epochs):
			model.train()
			y_train_prediction = model(x_train)
			loss = criterion(y_train_prediction, y_train)
			print(f"Epoch Training\t\t {t} {self.loss_function}  {loss.detach().item()}")
			hist_train[t] = loss.detach().item()
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()
			
			with torch.set_grad_enabled(False):
				model.eval()
				y_validation_prediction = model(x_validation)
				loss = criterion(y_validation_prediction,y_validation)
				print(f"Epoch Validation\t {t} {self.loss_function}  {loss.detach().item()}")
				hist_val[t] = loss.detach().item()

		training_time = time.time() - start_time
		print(f"Training time: {training_time}")

		model.eval()
		# make predictions
		y_validation_prediction = model(x_validation)

		# invert predictions
		y_train_prediction = labels_scaler.inverse_transform(y_train_prediction.detach().numpy())
		y_train = labels_scaler.inverse_transform(y_train.detach().numpy())
		y_validation_prediction = labels_scaler.inverse_transform(y_validation_prediction.detach().numpy())
		y_validation = labels_scaler.inverse_transform(y_validation.detach().numpy())


		train , val = error_calculation(self.loss_function, y_train, y_train_prediction, y_validation, y_validation_prediction)
		lstm = pd.DataFrame({'train': train, 'val' : val, 'time':training_time},index = [0])
		

		# plot_lstm(self.data,self.lookforward,y_train_prediction,y_validation_prediction,hist_train,hist_val) # Debug Function
		# save_df_to_db(pd.DataFrame(y_validation_prediction,),'lstmPrediction')
		hist = pd.DataFrame()
		hist['train'] = hist_train
		hist['val'] = hist_val
		# save_df_to_db(hist,'lstm_hist')
		# save_df_to_db(lstm,'lstm_metrics')
		return y_train_prediction.flatten().tolist(),hist,lstm
