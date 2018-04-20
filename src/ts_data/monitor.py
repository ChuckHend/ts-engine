import pandas as pd
import getStocks as gs
import utility.saveData
import googlefinance.client as gf
import sys

'''Usage:
python monitor.py <period>

where <period> is ("1d", "2d", "1w", "1m", etc)
'''

if len(sys.argv) > 1:
    period = str(sys.argv[1])
else:
    period = '1d'
    print('Retrieving prior {} of data, by default.'.format(period))


tickers=gs.get_tickers_index()
saveDir = '../data/30_second'

def save_stocks(data=tickers,saveDir=saveDir,interval="60",period=period):
    saveData.check_make_dir(saveDir)
    for ticker, dex in data:
        param = {'q': ticker,
                 'i': str(interval), # in seconds, 60 in the minimum
                 'x': dex,
                 'p': str(period)} # past
        df = pd.DataFrame(gf.get_price_data(param))
        maxDate = str(max(df.index)).replace(':','')
        minDate = str(min(df.index)).replace(':','')
        fname = '/{}_{}_{}.csv'.format(ticker, minDate, maxDate)
        df.to_csv(saveDir + fname)
        print('{} complete'.format(ticker))

def main():
    save_stocks(period=period)

if __name__ == "__main__":
    main()
