import pandas as pd
import torch

from data.training_data import training_data, training_data_no_missing_values
from models.KnnModel import KnnModel
from models.Linear import Linear

from models.LstmMVInput import LstmMVInput
from utils import utils

if __name__ == '__main__':

    # model_path = 'lstm_single_input'
    # lstm = LstmMVInput(loss_function=utils.MAE, data=training_data_no_missing_values(), model_path=model_path)
    # lstm = lstm.run_lstm()
    # torch.save(model, model_path)

    # knn = KnnModel(validation_size=0.2, features=pd.read_csv('FEATURES_USED.csv', index_col=0),
    #                  labels=pd.read_csv('SMP_VALUES.csv', index_col=0), n_neighbors_parameters=100)
    # knn.knn()

    linear = Linear(features=pd.read_csv('FEATURES_USED.csv', index_col=0), labels=pd.read_csv('SMP_VALUES.csv', index_col=0))
    linear.rum_linear()