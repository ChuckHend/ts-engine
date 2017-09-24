# -*- coding: utf-8 -*-

import os, errno, glob
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import bs4 as bs
import pickle
import requests
import csv




def saveStock(data, ticker):
    # check if directory exists
    saveDir='../data/{}'.format(ticker)
    if not os.path.exists(saveDir):
        try:
            os.makedirs(saveDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    today=dt.datetime.utcnow()
    fname='{}_{}.csv'.format(ticker, today.strftime('%d%m%Y'))             
    data.to_csv('{}/{}'.format(saveDir,fname), index=False)
    print('Successfully saved {}'.format(ticker.upper()))

def load_single(ticker):
    tickDir='../data/{}'.format(ticker)
    files=glob.glob('{}/*.csv'.format(tickDir))
    
    try:
        files=files[-1] #-1 to get largest dtstamp, most recent
        try:
            data=pd.read_csv(files)
            print('Loaded {}'.format(ticker.upper()))
            return data
        except IOError:
            print('Problem reading file: {}'.format(files))
            
    except IOError:
        print('No files found for {}'.format(ticker))
    
    
    
def get_single(ticker='AAPL', source='yahoo', save=True,
               start_date=dt.datetime(1995,1,1), end_date=dt.date.today()):
    print('Getting stock data for {} from {}...'.format(ticker.upper(), source))
    attempts=0
    while attempts <3:
            
        try:
            data = web.DataReader(ticker, source, start_date, end_date,)
            print('Successfully retrieved {}'.format(ticker.upper()))
            if save:
                saveStock(data,ticker)
            return data
        except RemoteDataError:
            attempts += 1
            print('Error retrieving {}: attempt {}'.format(ticker.upper(), attempts))
    

            
def save_tickers(returnTickers=False):
    saveDir='../data/tickers'
    today=dt.datetime.utcnow()
    if not os.path.exists(saveDir):
        try:
            os.makedirs(saveDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
  
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers =[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open('{}/sp500_{}.txt'.format(saveDir, today.strftime('%d%m%Y')), 'w') as myfile:
        wr = csv.writer(myfile,delimiter=',',quoting=csv.QUOTE_ALL)
        wr.writerow(tickers)

    if returnTickers:
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
        if (not os.path.exists('../data/{}.csv'.format(ticker))) or update_all:
            try:
                df = web.DataReader(ticker, source, start_date, end_date)
                df.to_csv('../data/{}.csv'.format(ticker))
                print('Success {}'.format(ticker))
            except RemoteDataError:
                print('ERROR')
        else:
            print('Already have {}'.format(ticker))
