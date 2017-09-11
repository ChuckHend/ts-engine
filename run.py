# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:30:04 2017

@author: hende
"""
import numpy as np
import featureEng as fe
import processStocks as ps
import getStocks as gs
import lstm 

# get a single stock's history (default is Apple)
data = gs.get_single()

# add derivative features
# data = fe.derivative(data, fill_na = True)

#normalise data
# TODO: find a better scaling method. should have -1:1 as range for tanh activation function
# Also, consider normalizing based on the sequence_length population rather than all data
data_n = ps.normalize_stock_data(data)
# data_n  = ps.z_score(data)

# training data
prediction_time = 1 # day, how far the window shifts?
testdatasize = 450
sequence_length = 50 # length of days in the prediction history
testdatacut = testdatasize + sequence_length  + 1

x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ]['Normalised Close'].as_matrix()
#y_train = data_n[prediction_time:-testdatacut  ]['Close'].as_matrix()

# test data
x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ]['Normalised Close'].as_matrix()
#y_test = data_n[prediction_time-testdatacut:  ]['Close'].as_matrix()


# unroll
x_train = ps.unroll(x_train,sequence_length)
x_test  = ps.unroll(x_test,sequence_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]

# build model architecture
model = lstm.build_model(x_train, timesteps=sequence_length, 
                         inlayer = 50, hidden1 = 100, 
                         hidden2=50, outLayer = 1)
                         
# train model
model.fit(
    x_train,y_train,
    batch_size=450, epochs=3,
    validation_split=0.20)


# plot results
predictions = lstm.predict_sequences_multiple(
        model, x_test, window_size=sequence_length, prediction_length=1)

predictions = lstm.predict_point_by_point(model, x_test)

predictions = lstm.predict_sequence_full(model, x_test, 
                                         window_size = sequence_length,)

lstm.plot_results_multiple(predictions, y_test, 50)


def plot_full(predicted_data, true_data, prediction_len):
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()





import sklearn.utils as sku

def mape(y_true, y_pred): 
    #mean absolute percentage error (MAPE)
    y_true, y_pred = sku.check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100   
                             

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, predictions)                         
                         
                         
                         
                         