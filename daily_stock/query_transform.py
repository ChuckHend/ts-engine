from sqlalchemy import *
import pandas as pd
import numpy as np
import random, sys

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
    #arch_engine = create_engine("postgresql://<user>:<pass>@<host>:5432/sampalytics")
    arch_engine = create_engine("postgresql://localhost:5432/sampalytics")

    # get list of tickers in our db
    tickers = arch_engine.engine.execute("SELECT DISTINCT ticker from stocks.minute").fetchall()
    # convert to list, ie get the tickers out of the tuples
    tickers = [x[0] for x in tickers]

    tickers = prune_list(outcome=outcome, ticks=tickers, n_predictors=n_predictors)

    # custom header
    full_df = pd.DataFrame(columns=['dtg'])
    # predictors = ['open', 'high', 'low', 'close', 'volume']
    predictors = ['close', 'volume'] # try less to save memory
    i = 1
    for ticker in tickers:
        header = ['dtg'] + [x + '_' + str(ticker) for x in predictors]
        select=', '.join(['dtg'] + predictors)
        # d = arch_engine.engine.execute("SELECT DISTINCT dtg, open, high, low, close, volume from stocks.minute WHERE ticker='{}'".format(ticker)).fetchall()
        d = arch_engine.engine.execute("SELECT DISTINCT {} from stocks.minute WHERE ticker='{}'".format(select, ticker)).fetchall()
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

    return full_df


def main():
    target, n_predictors = validateInput()

    df = query_db(outcome=target, n_predictors=n_predictors)

    df.to_csv('../data/{}_plus_{}.csv'.format(target, n_predictors),index=None)

    sys.exit()

if __name__ == "__main__":
    main()
