from ts_data.ts_data import ts_data as ts
import ts_models.ts_lstm as ts_lstm
import numpy as np
import pandas as pd
#import predicts
import matplotlib.pyplot as plt
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

n_in = 100
n_out = 100
target = 'sin'

# create sample data set (sin wave)
dataset = pd.DataFrame({'sin':[np.sin(i) for i in range(100000)]})
# view the last 100 data points in the sample set
# plt.plot(dataset[-100:])
# plt.legend()
# plt.show()

# pass the sin wave data to a time series object
ts_data = ts(n_in=n_in, n_out=n_out,target=target,rawData=dataset)
ts_data.roll_data()
ts_data.tscv(train=0.99)

# build the model
ts_model = ts_lstm.lstm_model(ts_data,
                              inlayer=int(ts_data.train_X.shape[-1])*2,
                              hiddenlayers=0,
                              loss_function='mean_squared_error',
                              dropout=0.05,
                              activation='tanh')

# fit the model
history = ts_model.fit(ts_data.train_X, ts_data.train_y,
                    epochs=15,
                    batch_size=1024,
                    validation_data=(ts_data.test_X, ts_data.test_y),
                    verbose=2,
                    shuffle=False)

# create new data to test the model
test = pd.DataFrame({'sin':[np.sin(i) for i in range(n_in + n_out)]})
ts_test = ts(n_in=n_in, n_out=n_out,target=target,rawData=test)
ts_test.roll_data()
ts_test.tscv(train=0.5)

yhat = ts_model.predict(ts_test.test_X)

x = [x for x in range(len(yhat[0]))]
plt.plot(x, ts_test.test_y[0], label='actual', )
plt.plot(x, yhat[0], label='predict')
plt.legend()
plt.show()
