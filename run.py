# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:30:04 2017

@author: hende
"""
import os
os.chdir('/Users/ahendel1/documents/academics/4cast')

import datetime as dt
import pandas_datareader
import numpy as np
import featureEng as fe
import processStocks as ps
import lstm 


#load data
start = dt.datetime(1995,1,1)
end   = dt.date.today()
data = pandas_datareader.data.DataReader('AAPL','yahoo',start,end)
data.head()

# add derivative features
data = fe.derivative(data, fill_na = True)

#normalise data
# TODO: find a better scaling method. should have -1:1 as range for tanh activation function
# Also, consider normalizing based on the sequence_length population rather than all data
data_n = ps.normalize_stock_data(data)
data_n  = ps.z_score(data)


# training data
prediction_time = 1 # day, how far the window shifts?
testdatasize = 450
sequence_length = 50 # length of days in the prediction history
testdatacut = testdatasize + sequence_length  + 1

x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ]['Normalised Close'].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ]['Close'].as_matrix()

# test data
x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ]['Normalised Close'].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ]['Close'].as_matrix()


# unroll
x_train = ps.unroll(x_train,sequence_length)
x_test  = ps.unroll(x_test,sequence_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]


print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


#build model architecture
model = lstm.build_model(x_train, timesteps=sequence_length, inlayer = 50,
                         hidden1 = 100, hidden2=50, outLayer = 1)
                         
#Step 3 Train the model
model.fit(
    x_train,
    y_train,
    batch_size=3028,
    epochs=1,
    validation_split=0.05)


#Step 4 - Plot the predictions!
# predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)

predictions = lstm.predict_point_by_point(model, x_test)

# lstm.plot_results_multiple(predictions, y_test, 50)

import sklearn.utils as sku


def mape(y_true, y_pred): 
    #mean absolute percentage error (MAPE)
    y_true, y_pred = sku.check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100   
                         

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, predictions)                         
                         
                         
                         
                         