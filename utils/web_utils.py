import plotly
import plotly.express as px
import pandas as pd

import json

def get_json_for_line_fig(df,x,y):
	fig = px.line(df,x=x,y=y)
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 


def get_json_for_fig_scatter(df,x,y):
	fig = px.scatter(df,x=x,y=y,trendline="ols",trendline_color_override='red')
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 

def get_metrics(model,db):
	metrics = db.get_data('*',f'metrics_{model}')
	metrics.index = pd.to_datetime(metrics.index)
	return metrics

	