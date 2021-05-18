import pandas as pd

from data.training_data import training_data
from models import lstm_model
from models.LstmSingleInput import SingleInputLstm

if __name__ == '__main__':

    #data = training_data()
    data = pd.read_csv('SMP_VALUES.csv')  # Offline Data
    price = data[['SMP']]

    model = SingleInputLstm(lstm_model.MSE, price, model_path='lstm_single_input')
    model.lstm()

