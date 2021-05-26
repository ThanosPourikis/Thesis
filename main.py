import pandas as pd
import torch

from data.training_data import training_data, training_data_no_missing_values
from models.KnnModel import KnnModel

from models.LstmSingleInput import LstmSingleInput
from utils import utils

if __name__ == '__main__':

    # model_path = 'lstm_single_input'
    # data =  # Offline Data Single Input
    # model = LstmSingleInput(loss_function=utils.MSE, data=training_data_no_missing_values(), model_path=model_path)
    # model = model.lstm()
    # torch.save(model, model_path)

    model = KnnModel(data=training_data(), validation_size=0.2)
    model.knn()