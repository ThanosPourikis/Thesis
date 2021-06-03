import pandas as pd

import flask
import sqlalchemy
from flask import render_template

from utils.utils import get_data, get_json_for_fig_line,get_json_for_fig_scatter

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config['Debug'] = True


@app.route('/')
def index():
	df = get_data('training_data','*')
	return render_template('home.jinja',title = 'Original Data',
							smp_json = get_json_for_fig_line(df,'Date','SMP'),
							res_total_json = get_json_for_fig_line(df,'Date','Res_Total'),
							load_total_json = get_json_for_fig_line(df,'Date','Load Total'),
							hydro_total_json = get_json_for_fig_line(df,'Date','Hydro Total'),
							sum_imports_json = get_json_for_fig_line(df,'Date','sum_imports'),
							sum_exports_json = get_json_for_fig_line(df,'Date','sum_exports'),
							)

@app.route('/Correlation')
def corrolations():
	df = get_data('training_data','*')
	return render_template('correlation.jinja',title = 'Correlation',
							res_total_json = get_json_for_fig_scatter(df,'Res_Total','SMP'),
							load_total_json = get_json_for_fig_scatter(df,'Load Total','SMP'),
							hydro_total_json = get_json_for_fig_scatter(df,'Hydro Total','SMP'),
							sum_imports_json = get_json_for_fig_scatter(df,'sum_imports','SMP'),
							sum_exports_json = get_json_for_fig_scatter(df,'sum_exports','SMP'),
							)

@app.route('/Linear')
def Linear():
	return 'Temp for Linear'

@app.route('/Knn')
def Knn():
	return 'Temp for Knn'


@app.route('/Lstm')
def lstm():
	return 'Temp for Lstm'


if __name__ == '__main__':
	app.run(host="localhost", port=8000, debug=True)

