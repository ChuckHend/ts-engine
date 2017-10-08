# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def plot_single(predicted, actual, ticker, data_set_category='test'):
    d = dt.datetime.today() - dt.timedelta(days = len(actual))
    d = d.year
    plt.title('{} Predicted Stock Daily Close ({})'.format(ticker.upper(), 
              data_set_category))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.ylabel('Daily Close')
    plt.xlabel('{} to present '.format(d))
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()

def stock_plot(data):
    # convert to tuple for plotting
    data = (data,)

    ax0 = plt.subplot2grid((6,2),(0,0),rowspan=5, colspan=1)
    ax1 = plt.subplot2grid((6,2),(5,0),rowspan=1, colspan=1, sharex=ax0)
    ax2 = plt.subplot2grid((6,2),(0,1),rowspan=5, colspan=1)
    ax3 = plt.subplot2grid((6,2),(5,1),rowspan=1, colspan=1, sharex=ax2)

    for each in data:
        ax0.plot(each.index,each['Adj Close'])
        ax1.plot(each.index,each['Adj Volume'])
        ax2.plot(each.index,each['Normalised Close'])
        ax3.plot(each.index,each['Normalised Volume'])

    plt.show()
    
def plot_full(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        # filling the array with 'none' so that no data is plotting in the trail
        padding = [None for p in range(i * prediction_len)]
        # use np.append
        plt.plot(np.insert(data, [0], padding), label='Prediction')
        plt.legend()
    plt.show()
    

    
def plot_features(dataset):
    features = list(dataset.columns)
    plt.figure()
    i = 1
    for feature in features:
        plt.subplot(len(features), 1, i)
        plt.plot(dataset[feature])
        plt.title(feature, y=0.5, loc='center')
        i +=1
    plt.show()
    
    
def plot_results_multiple(predicted_data, true_data, prediction_len, legend=True):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        if(legend):
            plt.legend()
    plt.show()
    
def plot_multiple(predicted_data, true_data,length):
    plt.plot(true_data.reshape(-1, 1)[length:])
    plt.plot(np.array(predicted_data).reshape(-1, 1)[length:])
    plt.show()