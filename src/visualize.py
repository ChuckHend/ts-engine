# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:57:32 2017

@author: hende
"""
import matplotlib.pyplot as plt

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
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
    