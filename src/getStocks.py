# -*- coding: utf-8 -*-

import os, errno, glob
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import googlefinance.client as gf
import bs4 as bs
import pickle
import requests
import csv
import random

def get_tickers_index(sectors = ['Information Technology', 'Energy']):
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    data.columns = [x.replace(' ','') for x in data.iloc[0,]]
    data.drop(0, axis=0, inplace=True)
    # append the index the ticker is listed on
    ind = pd.read_csv('tickers.csv',header=None)
    ind.columns=['Tickersymbol', 'dex']
    data=data.merge(ind, on='Tickersymbol')
    indices = [x in sectors for x in data.GICSSector]
    data = data[indices][['Tickersymbol', 'dex']]
    data=[tuple(x) for x in data.to_records(index=False)]
    return data


def saveStock(data, ticker):
    today=dt.datetime.utcnow()
    fname='{}_{}.csv'.format(ticker, today.strftime('%d%m%Y'))
    # check if directory exists
    saveDir='../data/{}'.format(ticker)
    if not os.path.exists(saveDir):
        try:
            os.makedirs(saveDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    data.to_csv('{}/{}'.format(saveDir,fname), index=False)
    print('Successfully saved {}'.format(ticker.upper()))

def load_single(ticker):
    tickDir='../data/{}'.format(ticker.upper())
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
        print('No files found for {}'.format(ticker.upper()))



def get_single(ticker='AAPL', source='yahoo', save=True,
               start_date=dt.datetime(1995,1,1), end_date=dt.date.today()):
    print('Getting stock data for {} from {}...'.format(ticker.upper(), source))
    attempts=0
    while attempts <3:

        try:
            data = web.DataReader(ticker, source, start_date, end_date,)
            data['Date'] = data.index

            print('Successfully retrieved {}'.format(ticker.upper()))
            if save:
                saveStock(data,ticker)
            return data
        except RemoteDataError:
            attempts += 1
            print('Error retrieving {}: attempt {}'.format(ticker.upper(), attempts))

def get_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers =[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    return tickers

def get_tickers_industry():
    exchanges = ['nasdaq', 'nyse', 'amex']
    df = pd.DataFrame()
    for ex in exchanges:
        print('Retrieving {}. . .'.format(ex))
        d = pd.read_csv('https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download'.format(ex))
        
        if ex=='nasdaq':
            d['Unnamed: 8']='NASD'
        else:
            d['Unnamed: 8']=ex.upper()
        d.rename(columns={'Unnamed: 8': 'exchange'}, inplace=True)
        df = pd.concat([df, d])
    return df

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
    '''iterates through S&P500, collects start_date to end_date stock data for each
    stock. saves each file as .csv to data/<ticker>/<ticker_ddmmyy.sv>'''
    #TODO: if a file from today's date exists, skip the file.

    if reload_sp500:
        tickers = save_tickers(returnTickers=True)
    else:
        with open("sp500tickers.pickle","wb") as f:
            tickers = pickle.load(f)

    today=dt.datetime.utcnow()

    for ticker in tickers:
        tickerDir='../data/{}'.format(ticker)
        fname='{}_{}.csv'.format(ticker, today.strftime('%d%m%Y'))
        print('{}\t: '.format(ticker), end="")

        if (not os.path.exists(tickerDir)) or update_all:
            try:
                df = web.DataReader(ticker, source, start_date, end_date)
                if not os.path.exists('../data/{}'.format(ticker)):
                    # create dir if it doesnt exists
                    os.makedirs(tickerDir)
                df.to_csv('../data/{}/{}'.format(ticker, fname))
                print('Success {}'.format(ticker))
            except RemoteDataError:
                print('ERROR')
        else:
            print('Already have {}'.format(ticker))

def saveScaled(data, n_in, n_out, ticker):
    today=dt.datetime.utcnow()
    fname='{}_{}_scaled_{}_{}.csv'.format(ticker, today.strftime('%d%m%Y'),
           n_in, n_out)
    # check if directory exists
    saveDir='../data/{}'.format(ticker)
    if not os.path.exists(saveDir):
        try:
            os.makedirs(saveDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    data.to_csv('{}/{}'.format(saveDir,fname), index=False)
    print('Successfully saved {}'.format(fname.upper()))

def latest_data(ticker):
    files=os.listdir('../data/{}/'.format(ticker))
    return files[-1]

def join_tgt_spt(target_ticker='UNH', rnd_spt=10, industry=None, exclude=None):
    '''joins supporting stocks to target stock data...supporting stocks are treated
    as additional features to target stock'''

    target_df=pd.read_csv('../data/{}/{}'.format(target_ticker,
        latest_data(target_ticker)),index_col=0)

    if industry:
        if not isinstance(industry, list):
            industry = list(industry)

        indPath = '../data/tickers/tickers_w_industry.csv'

        # read from file if exists, else get the updated list from web
        if os.path.exists(indPath):
            tickers = pd.read_csv(indPath, index_col=0)
        else:
            tickers = get_tickers_industry()

        # get the tickers in 'industry'
        tickers = tickers[[x in industry for x in tickers.industry]][['Symbol']]

        # remove target (already loaded and anything in our exlcude list)
        remove_list = list(target_ticker) + list(exclude)
        tickers = [x for x in tickers.Symbol if x not in remove_list]

        #tickers = tickers[tickers.Symbol != target_ticker].Symbol

    else: 
        tickers = os.listdir('../data')
        tickers.remove(target_ticker) # remove the targets folder name from the list
        tickers.remove('tickers') # remove tickers folder name from list
        tickers = random.sample(tickers, rnd_spt)

    for ticker in tickers:
        fpath = '../data/{}'.format(ticker)
        if os.path.exists(fpath):
            # load first df
            df=pd.read_csv('../data/{}/{}'.format(ticker,latest_data(ticker)),index_col=0)
            
            # merge on index, ie. date
            target_df=target_df.join(df, how='outer', rsuffix=ticker)

    target_df.dropna(axis=0, inplace=True)

    print('Join resulted in {} complete records.'.format(target_df.shape[0]))
    return target_df.reset_index()

def get_refresh(ticker_df,interval='1d',period='10y'):
    '''ticker_df is a dataframe with two columns. The first being the ticker and
    the second being the exchange or index'''
    data = [tuple(x) for x in ticker_df.to_records(index=False)]

    for ticker, dex in data:
        
        param = {'q': ticker,
                 'i': str(interval), # in seconds, 60 in the minimum
                 'x': dex,
                 'p': str(period)} # past
        print('Trying {}, {}'.format(ticker, dex))
        df = pd.DataFrame(gf.get_price_data(param))
        if df.shape[0] > 1:
            maxDate = str(max(df.index)).replace(':','')
            minDate = str(min(df.index)).replace(':','')
            fname = '/{}_{}_{}.csv'.format(ticker.upper(), minDate, maxDate)
            saveDir='../data/{}'.format(ticker)
            if not os.path.exists(saveDir):
                try:
                    os.makedirs(saveDir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            df.to_csv('{}/{}'.format(saveDir, fname))
        else:
            print('{} no data. . .'.format(ticker))

        print('Successfully saved {}'.format(ticker.upper()))
