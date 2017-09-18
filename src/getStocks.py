# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 07:49:28 2017

@author: hende
"""
import os
import datetime as dt
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import bs4 as bs
import pickle
import requests

# def select_data_source

def get_single(ticker='AAPL', source='yahoo', 
               start_date=dt.datetime(1995,1,1), end_date=dt.date.today()):
    print('Getting stock data for {} from {}...'.format(ticker, source))
    try:
        data = web.DataReader(ticker, source, start_date, end_date,)
        print('Successfully retrieved {}'.format(ticker))
        return data
    except RemoteDataError:
        print('ERROR...could not retrieve {}'.format(ticker))
            
        
def save_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers =[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
              pickle.dump(tickers,f)

    return tickers


def get_mkt_data(reload_sp500=True,update_all=True, source='yahoo',
                 start_date=dt.datetime(1970,1,1), end_date=dt.date.today()):
    if reload_sp500:
        tickers = save_tickers()
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