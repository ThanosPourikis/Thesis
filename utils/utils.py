import math

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from pytorch_forecasting import QuantileLoss
from models import lstm_model

MSE = 'MSE'
MAE = 'MAE'
HuberLoss = 'HuberLoss'
# QuantileLoss = 'QuantileLoss'


def error_calculation(function, y_train, y_train_prediction, y_test, y_test_prediction):

    if MAE == function:
        # calculate mean absolute error
        train_score = mean_absolute_error(y_train[:, 0], y_train_prediction[:, 0])
        print('Train Score: %.2f MAE' % train_score)
        test_score = mean_absolute_error(y_test[:, 0], y_test_prediction[:, 0])
        print('Test Score: %.2f MAE' % test_score)

    elif MSE == function:
        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(y_train[:, 0, ], y_train_prediction[:, 0,]))
        print('Train Score: %.2f RMSE' % train_score)
        test_score = math.sqrt(mean_squared_error(y_test[:, 0, ], y_test_prediction[:, 0,]))
        print('Test Score: %.2f RMSE' % test_score)
    return [train_score, test_score]


def loss_function_selection(function):
    if MAE == function:
        return torch.nn.L1Loss()
    elif MSE == function:
        return torch.nn.MSELoss(reduction='mean')
    elif HuberLoss == function:
        return torch.nn.SmoothL1Loss(reduction='mean')
    # elif QuantileLoss == function:
    #     return QuantileLoss()
