import os

import time
import datetime as dt

import numpy as np
import pandas as pd
from pandas_datareader import data#,wb
from pandas import DataFrame
from pandas import concat
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import matplotlib.pyplot as plt

import bs4 as bs
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers =[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
              pickle.dump(tickers,f)

    #print(tickers)

    return tickers

def get_mkt_data(reload_sp500=True,update_all=True, source='yahoo',
                 start_date=dt.datetime(1970,1,1), end_date=dt.date.today()):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","wb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
        print('{}\t: '.format(ticker), end="")
        if (not os.path.exists('stock_dfs/{}.csv'.format(ticker))) or update_all:
            try:
                df = web.DataReader(ticker, source, start_date, end_date)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
                print('Success'.format(ticker))
            except RemoteDataError:
                print('ERROR'.format(ticker))
        else:
            print('Already have'.format(ticker))


def normalize_stock_data(data):
    # ADJ data
    data_adj=data

    for i in range(0,data.index.shape[0]):
        data_adj.loc[data.index[i],'Ordinal/1e6'] = data.index[i].to_pydatetime().toordinal()/1e6
        data_adj.loc[data.index[i],'Weekday']     = data.index[i].to_pydatetime().weekday()

    # drop the non-normalized from new DF
    data_adj=data.drop(data.columns[[0,1,2,3,4,5]], axis=1)

    data_adj['Adj'] = data['Adj Close']/data['Close']

    data_adj['Adj Volume'] = data['Volume']
    #data_adj['Adj Volume'] -= np.min(data_adj['Adj Volume'])
    data_adj['Adj Volume'] /= np.max(data_adj['Adj Volume'])

    data_adj['Adj Close'] = data['Adj Close'] / data['Adj Close'][0]
    data_adj['Adj Open'] = data['Open']*data_adj['Adj'] / data['Adj Close'][0]
    data_adj['Adj High'] = data['High']*data_adj['Adj'] / data['Adj Close'][0]
    data_adj['Adj Low']  = data['Low'] *data_adj['Adj'] / data['Adj Close'][0]

    data_adj.loc[data.index[0],'Normalised Volume'] = 1
    data_adj.loc[data.index[1:],'Normalised Volume'] = data_adj['Adj Volume'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Volume'] -= 1

    data_adj.loc[data.index[0],'Normalised Close'] = 1
    data_adj.loc[data.index[1:],'Normalised Close'] = data_adj['Adj Close'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Close'] -= 1

    data_adj.loc[data.index[0],'Normalised Open'] = 1
    data_adj.loc[data.index[1:],'Normalised Open'] = data_adj['Adj Open'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Open'] -= 1

    data_adj.loc[data.index[0],'Normalised High'] = 1
    data_adj.loc[data.index[1:],'Normalised High'] = data_adj['Adj High'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised High'] -= 1

    data_adj.loc[data.index[0],'Normalised Low'] = 1
    data_adj.loc[data.index[1:],'Normalised Low'] = data_adj['Adj Low'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Low'] -= 1

    #reduce some mean
    data_adj=data_adj.drop(['Adj'], axis=1)

    return data_adj


def stock_plot(data):
    # convert to tuple for plotting
    data = (data,)

    ax0 = plt.subplot2grid((6,2),(0,0),rowspan=5, colspan=1)
    ax1 = plt.subplot2grid((6,2),(5,0),rowspan=1, colspan=1, sharex=ax0)
    ax2 = plt.subplot2grid((6,2),(0,1),rowspan=5, colspan=1)
    ax3 = plt.subplot2grid((6,2),(5,1),rowspan=1, colspan=1, sharex=ax2)

    for each in data:
        ax0.plot(each.index,each['Adj Close'])
        ax1.plot(each.index,each['Adj Volume'])
        ax2.plot(each.index,each['Normalised Close'])
        ax3.plot(each.index,each['Normalised Volume'])

    plt.show()

def single_batch(data_adj,pred_len=1):

    train_set_length = data_adj.shape[0]-(2*pred_len)+1
    train_set_width = data_adj.shape[1]

    train_X = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)
    train_Y = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)

    for i in range(0, train_set_length):
        start = i
        end =   i + pred_len
        train_X[i] = data_adj.ix[start:end].as_matrix()
        train_Y[i] = data_adj.ix[start+pred_len:end+pred_len].as_matrix()

    return train_X,train_Y

def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
