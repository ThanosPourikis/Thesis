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
    def __init__(self, loss_function, data, model_path,
                 learning_rate=0.001,
                 lookback=24,
                 input_dim=22,
                 hidden_dim=32,
                 num_layers=2,
                 output_dim=24,
                 num_epochs=100):

        self.loss_function = loss_function
        self.data = data
        self.lookback = lookback
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.learning_rate = learning_rate

    def run_lstm(self):

        scaler = MinMaxScaler(feature_range=(-1, 1))
        del self.data['Date']
        x_train, y_train, x_validation, y_validation = split_data(self.data, self.lookback, scaler)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_validation = torch.from_numpy(x_validation).type(torch.Tensor)

        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_validation_lstm = torch.from_numpy(y_validation).type(torch.Tensor)

        if path.exists(self.model_path):

            print('Found PreTrained Model  ..... \nLoading ..... ')
            model = torch.load(self.model_path)

        else:
            print('No preTrained Model Found ..... \nTraining .....')

            model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                         num_layers=self.num_layers)

        # choose loss function
        criterion = loss_function_selection(self.loss_function)

        optimiser = torch.optim.Adam(model.parameters(), self.learning_rate)

        hist = np.zeros(self.num_epochs)
        start_time = time.time()

        for t in range(self.num_epochs):
            y_train_prediction = model(x_train)
            loss = criterion(y_train_prediction, y_train_lstm)
            print(f"Epoch {t} {self.loss_function}  {loss.item()}")
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = time.time() - start_time
        print(f"Training time: {training_time}")

        model.eval()
        # make predictions
        y_validation_prediction = model(x_validation)

        # invert predictions
        y_train_prediction = scaler.inverse_transform(y_train_prediction.detach().numpy())
        y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
        y_validation_prediction = scaler.inverse_transform(y_validation_prediction.detach().numpy())
        y_validation = scaler.inverse_transform(y_validation_lstm.detach().numpy())

        lstm = error_calculation(self.loss_function, y_train, y_train_prediction, y_validation, y_validation_prediction)
        lstm.append(training_time)

        train_predict_plot = np.empty_like(self.data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.lookback:len(y_train_prediction) + self.lookback] = y_train_prediction[:]

        validation_predict_plot = np.empty_like(self.data)
        validation_predict_plot[:, :] = np.nan
        validation_predict_plot[len(y_train_prediction) + self.lookback - 2:len(self.data)-2] = y_validation_prediction[:]

        fig, axs = plt.subplots(2)

        axs[0].plot(train_predict_plot, color='r', label='Train Prediction')

        axs[0].plot(validation_predict_plot, color='b', label='Validation Prediction')
        axs[0].plot(self.data[-1], color='y', label='Actual Price')
        axs[0].set_title('Model')
        axs[0].set_xlim(len(y_train_prediction) - 25, len(y_train_prediction) + 50)
        # axs[0].set_xlim(len(self.data)-50, len(self.data)+100)
        axs[0].legend()
        axs[1].plot(hist, label='Loss')
        axs[1].set_title('Loss')
        plt.show()
        return model
