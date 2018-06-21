import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to disable GPU
from ts_data.ts_data import ts_data as ts
import getStocks
import visualize
import ts_models.predicts
import pandas as pd
import ts_models.ts_lstm as ts_lstm
import numpy as np

entityID = 'amd'.upper()
n_in = 360 # number of timesteps as input
n_out = 360 # number of timesteps as output
target = 'close_AMD'

dataset = pd.read_csv('/path/AMD_plus_100.csv')

## might need to remove these if its already in data prep
# ie. row 1 is Jan 1, row 2 in Jan 2.
dataset.sort_values('dtg', inplace=True)
# and do a forward fill
dataset.fillna(method='ffill', inplace=True)
# then drop the NA
dataset.dropna(inplace=True)
# rename dtg to date for TS package
dataset.rename(columns={'dtg':'Date'}, inplace=True)
###

ts_data = ts(n_in=n_in, 
             n_out=n_out, 
             entityID=entityID,
             target=target,
             rawData=dataset)

ts_data.eng_features(derivate=False, weekdays=False)

ts_data.roll_data()

ts_data.data.to_csv('AMD_TS_DATA.csv',index=None)

ts_data.tscv(train=0.98)

ts_data.data.tail()

ts_model = ts_lstm.lstm_model(ts_data, 
                               inlayer=int(ts_data.train_X.shape[-1])*2,
                               hiddenlayers=0,
                               loss_function='mae',
                               dropout=0.05,
                               activation='tanh',
                               gpus=1)

import time
start = time.time()
history = ts_model.fit(ts_data.train_X, ts_data.train_y, 
                    epochs=50, 
                    batch_size=1024, 
                    validation_data=(ts_data.test_X, ts_data.test_y), 
                    verbose=2, 
                    shuffle=False)
fitTime = time.time()-start
print('Fit Time: {}'.format(round(fitTime,2)))

# select the last sequence of data (it is length of n_in)
test_X = ts_data.test_X[-1]
test_Y = ts_data.test_y[-1]

yhat = ts_model.predict(test_X.reshape(1, test_X.shape[-2], test_X.shape[-1]))

def mse(y, yhat):
    return np.mean((y=yhat)**2)

error_metric = mse(test_Y, yhat)

print('MSE: {}'.format(round(error_metric, 4)))