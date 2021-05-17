import numpy as np
import pandas as pd

from models import lstm_model
from models.Lstm_run import RunLstm

if __name__ == '__main__':

    # data = training_data()
    data = pd.read_csv('SMP_VALUES.csv')  # Offline Data
    price = data[['SMP']]

    model = RunLstm(lstm_model.MSE, price)
    plt = model.lstm()

    plt.plot(price['SMP'])
    plt.show()
