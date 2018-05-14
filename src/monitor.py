import pandas as pd
import getStocks as gs
import json
import datetime
import utility.saveData as saveData
import googlefinance.client as gf
import sys, os

'''Usage:
python monitor.py <period>

where <period> is ("1d", "2d", "1w", "1m", etc)
'''

if len(sys.argv) > 1:
    period = str(sys.argv[1])
else:
    period = '1d'
    print('Retrieving prior {} of data, by default.'.format(period))

with open('stock_config.json') as f:
    config = json.loads(f.read())

if 'sector' in config.keys():
    sector = config['sector']
else:
    sector='all'

if 'industry' in config.keys():
    industry = config['industry']
else:
    industry='all'

tickers=gs.get_tickers_industry_sector(sector=sector, industry=industry)
tickers=tickers[['Symbol','exchange']]
tickers=[tuple(x) for x in tickers.to_records(index=False)]
saveDir = '../data/60_second'

print('Fetching\n Sectors: {} n\ Industries: {}'.format(sector, industry))

def save_stocks(data=tickers,saveDir=saveDir,interval="60",period=period):
    saveData.check_make_dir(saveDir)
    for ticker, dex in data:
        param = {'q': ticker,
                 'i': str(interval), # in seconds, 60 in the minimum
                 'x': dex,
                 'p': str(period)} # past
        df = pd.DataFrame(gf.get_price_data(param))

        recs = df.shape[0]

        if recs > 0:
            maxDate = str(max(df.index)).replace(':','')
            minDate = str(min(df.index)).replace(':','')
            fname = '/{}_{}_{}.csv'.format(ticker, minDate, maxDate)
            df.to_csv(saveDir + fname)
            outStr = '{} complete {} records'.format(ticker, recs)
            print(outStr)
            log(outStr)
        else:
            outStr = 'No records for {}'.format(ticker)
            print(outStr)
            log(outStr)


def log(log_entry):
    fname = 'monitor.log'
    
    if os.path.exists(fname):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    f = open(fname, append_write)

    time = str(datetime.datetime.now()).split('.')[0]

    f.write(time + ',' + str(log_entry) + '\n')
    f.close()


def main():
    save_stocks(data=tickers, period=period)


if __name__ == "__main__":
    main()
