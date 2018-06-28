from sqlalchemy import *
import pandas as pd
import numpy as np
import random, sys
import os
import ts_config as cfg

# queries archbox or sql db for data
# transforms the stock data from long to wide
# each row becomes a single date/time
# columns are created for all the stocks

# call this script from command line
# saves a transformed dataframe in .csv at the end of main()
# to ../../data/<target>_plus_<n_predictors>.csv

def validateInput():
    if len(sys.argv) != 3:
        print('[ERROR] USAGE: python query_transform.py <target> <n_predictors>')
        sys.exit()
    else:
        return(str(sys.argv[1]), int(sys.argv[2]))     


def prune_list(outcome, ticks, n_predictors):
    # remove the outcome var so we dont randomly select it and create duplicates
    ticks = [x for x in ticks if x not in outcome] # .remove causes pointer issue
    # convert list of tickers to df so we can index
    df = pd.DataFrame(ticks)
    # randomly select the rows we want to take
    random.seed(111)
    rows = random.sample(range(len(ticks)), n_predictors)
    # then select the rows subset
    df = df.iloc[rows]
    return [outcome] + list(df[0])

def query_db(outcome, n_predictors):
    # setup engine to archbox psql database
    #arch_engine = create_engine("postgresql://<user>:<pass>@192.168.0.2:5432/sampalytics")
    # arch_engine = create_engine("postgresql://localhost:5432/sampalytics")
    user = str(cfg.load_config('username'))
    passwd = str(cfg.load_config('password'))
    host = cfg.load_config('host')
    port = cfg.load_config('port')
    database = cfg.load_config('database')
    schema_table = cfg.load_config('schema.table')
    predictor_vars = cfg.load_config('predictor_variables')

    debug = cfg.load_config('debug')
    connection_str = 'postgresql://{}:{}@{}:{}/{}'.format(user,passwd,host,port,database)
    arch_engine = create_engine(connection_str)

    # get list of tickers in our db
    tickers = arch_engine.engine.execute('SELECT DISTINCT ticker from {}'.format(schema_table)).fetchall()
    # convert to list, ie get the tickers out of the tuples
    tickers = [x[0] for x in tickers]

    tickers = prune_list(outcome=outcome, ticks=tickers, n_predictors=n_predictors)

    # custom header
    full_df = pd.DataFrame(columns=['dtg'])
    i = 1
    for ticker in tickers:
        header = ['dtg'] + [x + '_' + str(ticker) for x in predictor_vars]
        select = ', '.join(['dtg'] + predictor_vars)
        # d = arch_engine.engine.execute("SELECT DISTINCT dtg, open, high, low, close, volume from stocks.minute WHERE ticker='{}'".format(ticker)).fetchall()
        d = arch_engine.engine.execute(
            "SELECT DISTINCT {} from stocks.minute WHERE ticker='{}'".format(
                select, 
                ticker)).fetchall()
        d = pd.DataFrame(d, columns=header).dropna()
        print('Loading: {}: {}'.format(ticker, i))
        i = i + 1
        full_df = pd.merge(full_df, d, on='dtg', how='outer')

    # ie. row 1 is Jan 1, row 2 in Jan 2.
    full_df.sort_values('dtg', inplace=True)
    # and do a forward fill
    full_df.fillna(method='ffill', inplace=True)
    # then drop the NA
    full_df.dropna(inplace=True)
    # rename dtg to date for TS package
    full_df.rename(columns={'dtg':'Date'}, inplace=True)

    if debug:
        # if debug/test mode, limit to 10k rows
        full_df = full_df.tail(10000)

    return full_df

def check_make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def main():
    # target, n_predictors = validateInput()
    
    target = cfg.load_config('target')
    n_predictors = cfg.load_config('n_pred')
    workdir = cfg.load_config('workdir')

    df = query_db(outcome=target, n_predictors=n_predictors)

    # make the '../data' directory if it does not exist
    check_make_dir('./../data/')
    df.to_csv('{}/../data/{}_plus_{}.csv'.format(workdir, target, n_predictors),index=None)

    sys.exit()

if __name__ == "__main__":
    main()
