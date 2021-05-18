import math
import time
from os import path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from models import lstm_model

from models.lstm_model import LSTM
from models.sliding_windows import split_data


def error_calculation(function, y_train, y_train_prediction, y_test, y_test_prediction):

    if lstm_model.MAE == function:
        # calculate mean absolute error
        train_score = mean_absolute_error(y_train[:, 0], y_train_prediction[:, 0])
        print('Train Score: %.2f MAE' % train_score)
        test_score = mean_absolute_error(y_test[:, 0], y_test_prediction[:, 0])
        print('Test Score: %.2f MAE' % test_score)

    else:
        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(y_train[:, 0], y_train_prediction[:, 0]))
        print('Train Score: %.2f RMSE' % train_score)
        test_score = math.sqrt(mean_squared_error(y_test[:, 0], y_test_prediction[:, 0]))
        print('Test Score: %.2f RMSE' % test_score)
    return [train_score, test_score]


def loss_function_selection(function):
    if lstm_model.MAE == function:
        return torch.nn.L1Loss()
    elif lstm_model.MSE == function:
        return torch.nn.MSELoss(reduction='mean')
    elif lstm_model.HuberLoss == function:
        return torch.nn.SmoothL1Loss(reduction='mean')
    elif lstm_model.QuantileLoss == function:
        return torch.nn.QuantileLoss


class SingleInputLstm:
    def __init__(self, loss_function, price,
                 lookback=24,
                 input_dim=1,
                 hidden_dim=32,
                 num_layers=1,
                 output_dim=1,
                 num_epochs=100, model_path=None):

        self.loss_function = loss_function
        self.price = price
        self.lookback = lookback
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.model_path = model_path

    def lstm(self):

        sc = MinMaxScaler()
        price = sc.fit_transform(self.price)
        x_train, y_train, x_test, y_test = split_data(price, self.lookback)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)

        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        if path.exists(self.model_path):

            print('Found PreTrained Model  ..... \nLoading ..... ')
            model = torch.load(self.model_path)

        else:
            print('No preTrained Model Found ..... \nTraining .....')

            model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                         num_layers=self.num_layers)

        # chose loss function
        criterion = loss_function_selection(self.loss_function)

        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        hist = np.zeros(self.num_epochs)
        start_time = time.time()

        for t in range(self.num_epochs):
            y_train_prediction = model(x_train)
            loss = criterion(y_train_prediction, y_train_lstm)
            print(f"Epoch {t} {self.loss_function} ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        model.eval()
        # make predictions
        y_test_prediction = model(x_test)
        # invert predictions
        y_train_prediction = sc.inverse_transform(y_train_prediction.detach().numpy())
        y_train = sc.inverse_transform(y_train_lstm.detach().numpy())
        y_test_prediction = sc.inverse_transform(y_test_prediction.detach().numpy())
        y_test = sc.inverse_transform(y_test_lstm.detach().numpy())

        lstm = error_calculation(self.loss_function, y_train, y_train_prediction, y_test, y_test_prediction)
        lstm.append(training_time)

        train_predict_plot = np.empty_like(price)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.lookback:len(y_train_prediction) + self.lookback] = y_train_prediction[:]

        test_predict_plot = np.empty_like(price)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(y_train_prediction) + self.lookback:len(price)] = y_test_prediction

        plt.plot(train_predict_plot, color='r', label='Train Prediction')

        plt.plot(test_predict_plot, color='b', label='Test Prediction')
        plt.plot(self.price, color='y', label='Actual Price')
        plt.xlim(3350, 3450)
        plt.legend()

        torch.save(model, f'lstm_single_input')
        plt.show()
