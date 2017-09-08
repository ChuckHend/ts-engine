# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:30:04 2017

@author: hende
"""

import time
import datetime as dt
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import processStocks as sdp
import pandas
import pandas_datareader
import featureEng as fe
import processStocks as ps


#load data
start = dt.datetime(1995,1,1)
end   = dt.date.today()
data = pandas_datareader.data.DataReader('ASTC','yahoo',start,end)
data.head()

# add derivative features
data = fe.derivative(data, fill_na = True)

#normalise data
data_n = sdp.normalize_stock_data(data)

# training data
prediction_time = 1 # day, how far the window shifts?
testdatasize = 450
sequence_length = 50 # length of days in the prediction history
testdatacut = testdatasize + sequence_length  + 1

x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ]['Normalised Close'].as_matrix()

# test data
x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ]['Normalised Close'].as_matrix()

# unroll
x_train = ps.unroll(x_train,sequence_length)
x_test  = ps.unroll(x_test,sequence_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]


print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm 

#Step 2 Build Model ### replace with function
model = Sequential()

model.add(LSTM(
    input_shape=(None, x_train.shape[-1]),
    units =50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    units=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : {}'.format(time.time() - start))

# assign the network architecture
# timesteps should be the 'window' or 'sequence'?
arch = lstm.neuralArch(hidden_layer = 50, outlayer = 1, 
                       timesteps = 1,  x_train = x_train)


#Step 3 Train the model
model.fit(
    x_train,
    y_train,
    batch_size=3028,
    epochs=50,
    validation_split=0.05)


#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)

