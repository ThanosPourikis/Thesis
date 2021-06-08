import time
from os import path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from models.lstm_model import LSTM
from data.sliding_windows import split_data
from utils.utils import loss_function_selection, error_calculation


class LstmMVInput:
	def __init__(self, loss_function, data,
				 learning_rate=0.001,
				 lookback=24,
				 input_dim=6,
				 hidden_dim=32,
				 num_layers=2,
				 output_dim=24,
				 num_epochs=500):

		self.loss_function = loss_function
		self.data = data
		self.lookback = lookback
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.output_dim = output_dim
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate

	def run_lstm(self):
		del self.data['Date']

		scaler = MinMaxScaler(feature_range=(-1, 1))
		labels_scaler = MinMaxScaler(feature_range=(-1, 1))
		x_train, y_train, x_validation, y_validation = split_data(self.data, self.lookback)

		x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
		x_validation = scaler.transform(x_validation.reshape(-1, x_validation.shape[-1])).reshape(x_validation.shape)
		y_train = labels_scaler.fit_transform(y_train)
		y_validation = labels_scaler.transform(y_validation)

		x_train = torch.from_numpy(x_train).type(torch.Tensor)
		x_validation = torch.from_numpy(x_validation).type(torch.Tensor)

		y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
		y_validation_lstm = torch.from_numpy(y_validation).type(torch.Tensor)

		print('No preTrained Model Found ..... \nTraining .....')

		model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
						num_layers=self.num_layers)

		# choose loss function
		criterion = loss_function_selection(self.loss_function)

		optimiser = torch.optim.Adam(model.parameters(), self.learning_rate)

		hist_train = np.zeros(self.num_epochs)
		hist_val = np.zeros(self.num_epochs)

		start_time = time.time()

		for t in range(self.num_epochs):
			y_train_prediction = model(x_train)
			loss = criterion(y_train_prediction, y_train_lstm)
			print(f"Epoch {t} {self.loss_function}  {loss.item()}")
			hist_train[t] = loss.item()
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()


		training_time = time.time() - start_time
		print(f"Training time: {training_time}")

		model.eval()
		# make predictions
		y_validation_prediction = model(x_validation)

		# invert predictions
		y_train_prediction = labels_scaler.inverse_transform(y_train_prediction.detach().numpy())
		y_train = labels_scaler.inverse_transform(y_train_lstm.detach().numpy())
		y_validation_prediction = labels_scaler.inverse_transform(y_validation_prediction.detach().numpy())
		y_validation = labels_scaler.inverse_transform(y_validation_lstm.detach().numpy())

		lstm = error_calculation(self.loss_function, y_train, y_train_prediction, y_validation, y_validation_prediction)
		lstm.append(training_time)

		train_predict_plot = np.empty_like(self.data.iloc[:, -1])
		train_predict_plot[:] = np.nan
		train_predict_plot[self.lookback:len(y_train_prediction.flatten()) + self.lookback] = y_train_prediction.flatten()

		validation_predict_plot = np.empty_like(self.data.iloc[:, -1])
		validation_predict_plot[:] = np.nan
		validation_predict_plot[len(y_train_prediction.flatten()) + self.lookback +9 : len(self.data) ] = y_validation_prediction.flatten()

		fig, axs = plt.subplots(2)

		axs[0].plot(train_predict_plot, color='r', label='Train Prediction')

		axs[0].plot(validation_predict_plot, color='b', label='Validation Prediction')
		axs[0].plot(self.data.iloc[:, -1], color='y', label='Actual Price')
		axs[0].set_title('Model')
		# axs[0].set_xlim(len(y_train_prediction) - 25, len(y_train_prediction) + 50)
		axs[0].set_xlim(len(self.data)-50, len(self.data)+100)
		axs[0].legend()
		axs[1].plot(hist_train, label='Loss')
		axs[1].plot(hist_val)
		axs[1].set_title('Loss')
		plt.show()
		return 0
