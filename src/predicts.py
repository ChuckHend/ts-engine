# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:54:54 2017

@author: hende
"""
import numpy as np
from numpy import newaxis


def onlineForecast(model, X, y, n_in=1, batch_size=1):
    for i in range(len(X)):
        testX, testy = X[i], y[i]
        testX = testX.reshape(1, n_in, X.shape[-1]) # (rows, n_in, features)
        yhat = model.predict(testX, batch_size=batch_size)
        print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
        
def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, 
    # only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs