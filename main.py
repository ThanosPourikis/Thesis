import torch
from utils.database_interface import DB
from update_data import update
from data.get_weather_data import get_weather_data
from data.units_data import get_unit_data
from data.get_SMP_data import get_SMP_data
from math import nan
from models.XgB import XgbModel
import numpy as np
from sklearn import utils
from models.lstm.LstmMVInput import LstmMVInput
import pandas as pd
from models.KnnModel import KnnModel
from models.Linear import Linear
from data.ADMHE_files import get_excel_data
from utils import utils
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go



# db = DB()
# df = db.get_data('dataset','*')
df = pd.read_csv('training_set.csv')

# linear = Linear(data=df)
# prediction, train_score, validation_score = linear.train()
# fig = px.line(prediction,x='Date',y=['Prediction','SMP'])
# fig = fig.update_xaxes(rangeslider_visible=True)
# fig.show()

lstm = LstmMVInput(utils.MAE,df,num_epochs=150,batch_size=24,sequence_length=24)
y_train_pred,hist,model = lstm.train()
fig = go.Figure()
fig.add_trace(go.Scatter(
	x = hist.index,
	y=hist['val'],
	mode = 'lines+markers'
))
fig.add_trace(go.Scatter(
	x = hist.index,
	y=hist['train'],
	mode = 'lines+markers'
))
fig.show()
torch.save(model,'lstm.model')
