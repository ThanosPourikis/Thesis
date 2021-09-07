import plotly
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

import json

def get_json_for_line_fig(df,x,y):
	fig = px.line(df,x=x,y=y)
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_json_for_line_scatter(df,y,line = None):
	fig = go.Figure()
	for i in y:
		fig.add_trace(go.Scatter(x=df.index, y=df[i],
					mode='lines+markers',
					name=df[i].name))
		if line != None:
			fig.add_vline(x=line,line_dash="dash", line_color="purple",annotation_text = 'Best Epoch')
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 


def get_json_for_fig_scatter(df,y,x):
	fig = px.scatter(df,x=x,y=y,trendline="ols",trendline_color_override='red')
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 

def get_metrics(model,db):
	metrics = db.get_data('*',f'metrics_{model}')
	# metrics.index = pd.to_datetime(metrics.index)
	return metrics


def get_candlesticks(df):
	export = pd.DataFrame()
	# export['High'] = [df[i:i+24].max() for i in range(0,len(df),24)]
	for i in range(0,len(df),24):
		temp = pd.DataFrame()
		temp['Date'] = [df[i:i+24].index[0]]
		temp['high'] = df[i:i+24].max()
		temp['low'] = df[i:i+24].min()
		temp['open'] = df[i:i+24].iloc[0]
		temp['close'] = df[i:i+24].iloc[-1]
		temp = temp.set_index('Date')
		export = pd.concat([export,temp])

	fig = go.Figure()
	fig = go.Figure(data=[go.Candlestick(x=export.index,
				open=export['open'], high=export['high'],
				low=export['low'], close=export['close'])
					 ])
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_heatmap(df):
	units_24 = df.copy()
	units_24 = pd.DataFrame([units_24.iloc[i] for i in range(0,units_24.shape[0],24)])
	units_24.index = [units_24.index[i].date() for i in range(units_24.shape[0])]
	units_24.head()
	units_24_trans = units_24.iloc[:,:-1].transpose()

	heatmap = go.Heatmap(
		z = units_24_trans.values,
		x= units_24_trans.columns,
		y = units_24_trans.index
	)
	fig = go.Figure(data=[heatmap])
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

