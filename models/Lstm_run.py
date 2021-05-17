import math
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from models import lstm_model

from models.lstm_model import LSTM
from models.sliding_windows import split_data


class RunLstm:
    def __init__(self, loss_function, price,
                 lookback=24,
                 input_dim=1,
                 hidden_dim=32,
                 num_layers=2,
                 output_dim=1,
                 num_epochs=100):

        self.loss_function = loss_function
        self.price = price
        self.lookback = lookback
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_epochs = num_epochs

    def lstm(self):

        sc = MinMaxScaler()
        price = sc.fit_transform(self.price)
        x_train, y_train, x_test, y_test = split_data(price, self.lookback)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)

        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                     num_layers=self.num_layers)

        # chose loss function
        if lstm_model.MAE == self.loss_function:
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss(reduction='mean')

        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.num_epochs)
        start_time = time.time()

        for t in range(self.num_epochs):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train_lstm)
            print(f"Epoch {t} {self.loss_function} ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))



        # make predictions
        y_test_pred = model(x_test)
        lstm = []
        # invert predictions
        y_train_pred = sc.inverse_transform(y_train_pred.detach().numpy())
        y_train = sc.inverse_transform(y_train_lstm.detach().numpy())
        y_test_pred = sc.inverse_transform(y_test_pred.detach().numpy())
        y_test = sc.inverse_transform(y_test_lstm.detach().numpy())

        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % train_score)
        test_score = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % test_score)
        lstm.append(train_score)
        lstm.append(test_score)
        lstm.append(training_time)

        train_predict_plot = np.empty_like(price)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.lookback:len(y_train_pred) + self.lookback, :] = y_train_pred

        test_predict_plot = np.empty_like(price)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(y_train_pred) + self.lookback - 1:len(price) - 1, :] = y_test_pred

        plt.plot(train_predict_plot, color='r', label='Prediction')

        plt.plot(test_predict_plot, color='b', label='Prediction')

        original = sc.inverse_transform(price)
        plt.plot(original, color='y')
        plt.xlim(3200,3500)
        plt.show()
