import glob


fdir = '../data/30_second/'

files = glob.glob(fdir + '*')

def strip_ticker(s):
    return s.strip(fdir).split('_')[0]

tickers = list(set([strip_ticker(x) for x in files]))

tickers = {key:[] for key in tickers}

for file in files:
    tickers[strip_ticker(file)].append(file)
