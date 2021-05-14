import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import LSTM
from models.sliding_windows import split_data


def run_lstm():
    lookback = 20

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    # data = training_data()
    data = pd.read_csv('data.csv')  # Offline Data
    price = data[['SMP']]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['SMP'] = scaler.fit_transform(price['SMP'].values.reshape(-1, 1))

    x_train, y_train, x_test, y_test = split_data(price, lookback)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)

    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    print("Mean Absolute Error{}".format(hist.mean()))
    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

    plt.plot(predict, color='r', label='Prediction')
    plt.plot(original, color='b', label='Original')
    plt.plot(data["SMP"], color='g', label='Actual')
    plt.xlim(3200, 3500)
    plt.show()
