import pandas as pd
import torch

from data.training_data import training_data
from models import lstm_model
from models.LstmSingleInput import LstmSingleInput

if __name__ == '__main__':

    data = training_data()

    # data = pd.read_csv('data.csv')  # Offline Data MV input
    # models = LstmMvInput(data[:4211], model_path='lstm_mv_input')
    # models.lstm()

    model_path = 'lstm_single_input'
    data = pd.read_csv('SMP_VALUES.csv')  # Offline Data Single Input
    data = data[['SMP']]
    model = LstmSingleInput(lstm_model.MSE, data, model_path)
    model = model.lstm()
    #torch.save(model, model_path)
