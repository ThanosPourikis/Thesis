import pandas as pd

from models import lstm_model
from models.LstmSingleInput import RunSingleInputLstm

if __name__ == '__main__':

    # data = training_data()
    data = pd.read_csv('SMP_VALUES.csv')  # Offline Data
    price = data[['SMP']]

    model = RunSingleInputLstm(lstm_model.MSE, price, model_path='lstm_single_input')
    model.lstm()

