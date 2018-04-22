from ts_data.ts_data import ts_data as ts
import getStocks
import visualize
import ts_models.predicts
import pandas as pd
import ts_models.ts_lstm as ts_lstm
import time

ticker = 'amd'.upper()
n_in = 100 # number of historical days to input
n_out = 10 # days into future to predict
target = 'd1close'

dataset = getStocks.join_tgt_spt(
    target_ticker=ticker.upper(), 
    industry=['Semiconductors'], 
    exclude=['MU','QRVO'])

dataset.rename(columns=lambda x: x.replace(' ',''),inplace=True)

ts_data = ts(n_in=n_in, 
             n_out=n_out, 
             ticker=ticker,
             target=target,
             rawData=dataset)

ts_data.eng_features(derivate=True, weekdays=False)

ts_data.roll_data()

ts_data.tscv(train=0.98)

ts_model = ts_lstm.lstm_model(ts_data, 
                               inlayer=int(ts_data.train_X.shape[-1])*2,
                               hiddenlayers=0,
                               loss_function='mean_squared_error',
                               dropout=0.05,
                               activation='tanh',
                               gpus=1)


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

import matplotlib.pyplot as plt
x = [x for x in range(len(yhat[0]))]
plt.plot(x, yhat[0], label='predicted change in price')
plt.plot(x, test_Y, label='actual change in price')
plt.title(ticker)
plt.xlabel('days')
plt.ylabel('amount change in price')
plt.legend()
plt.show()
