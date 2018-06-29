# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import json
from dash.dependencies import Input, Output, State
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

app = dash.Dash()

with open('defaults/config_default.json','r') as f:
    config_default = json.load(f)

app.layout = html.Div([
    html.H1(
        children='forecast-engine',
        style={
        'display' : 'inline',
        'float': 'left'},
        className = 'topRow'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),

    html.Label('Input ticker'),
    dcc.Input(
        id='ticker',
        placeholder='Enter a value...',
        type='text',
        value=''
    ),

    html.Button('Predict', id='predict'),

    # place holder for output from predict
    html.Div(id='output')




], style={'columnCount': 1})


@app.callback(
    Output('output', 'children'),
    [Input('ticker','value'),
     Input('predict', 'n_clicks')])
def predict(ticker, clicks):

    if clicks:
        new_config = {'target' : ticker}
        save_config(new_config)
        subprocess.call(['python3', 'main.py'])
        results = pd.read_csv('../data/{}_results.csv'.format(ticker), header=None)

        fig = dcc.Graph(id='output',
            figure = {
            'data': [
                {'y': results[0], 'type': 'line'}]
            })

        return fig



def save_config(key_values):
    print(key_values)
    for key, value in key_values.items():
        config_default[key] = value
    with open('config/config.json','w') as out:
        json.dump(config_default, out, indent=4)


if __name__ == '__main__':
    app.run_server(debug=True)