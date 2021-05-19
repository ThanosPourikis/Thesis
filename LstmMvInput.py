from os import path

import torch.utils.data

from models.lstm_model import LSTM
from models.sliding_windows import split_data


class LstmMvInput:
    def __init__(self, data, model_path=None,
                 lookback=24,
                 input_dim=6,
                 hidden_dim=32,
                 num_layers=1,
                 output_dim=1,
                 num_epochs=100):
        self.data = data
        self.lookback = lookback
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.model_path = model_path

    def lstm(self):

        x_train, y_train, x_test, y_test = split_data(data=self.data, lookback=self.lookback)

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

        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        for i in range(self.num_epochs):
            output = model(x_train)
            #loss = criterion(output, y_test_lstm)
            optimiser.zero_grad()
            #loss.backwards()
            optimiser.step()

        model.eval()
        # make predictions
        y_test_prediction = model(x_test)
        print(y_test_prediction)
