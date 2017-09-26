# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import sys
import numpy as np
from pandas import DataFrame
from pandas import concat
import pandas as pd


def frame_targets(dataset, features, n_out,target='Close'):
    # create the list of features we dont want to predict
    # so take all features, less the one we want to predict
    dropList = list(features)
    dropList.remove(target)
    
    # iterate over droplist, removing all columns in the droplist
    # for t....t+n...which will be t+n_in-1
    for feature in dropList:
        # iterate over n_out-1
        dataset.drop('{}(t)'.format(feature), axis=1, inplace=True)
        
        # drop all t+ that arent our target feature
        for i in range(1,n_out):
            tfeat='{}(t+{})'.format(feature,i)
            dataset.drop(tfeat, axis=1, inplace=True)
            
    return dataset



def get_filter_seq(features, n_in, n_out):
    # creates the vector of columns we want from the outer scaled dataset
    # outer being a set of t-n to t+n
    featStrings=[]
    for feat in features:
        featStrings.append('{}(t)'.format(feat))
        
        for x_in in range(1,n_in+1):
            featStrings.append('{}(t-{})'.format(feat, x_in))
            
        for x_out in range(1,n_out): # t+1 would be n_out=2
            featStrings.append('{}(t+{})'.format(feat, x_out))
    return featStrings


def scale_sequence(dataset, features):
    # accepts a reframed dataset (already in supervised time series)
    # scaled each observation, based on the data in the observation
    # on a feature by feature basis
    # features are the global feature set(prior to time series conv.)
    # use normalise window form instead of monmax
    
    #TODO: reassign derivative features to percentage change in fe fun
    features=features.drop(['d1close', 'd2close'])
    
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    
    scaled=pd.DataFrame(columns=dataset.columns)
    
    for row in range(dataset.shape[0]):
        d=dataset.iloc[row,:]
        d=pd.DataFrame(d.values.reshape(1,d.shape[0]), columns=dataset.columns)
        # iterate over each feature
        # scaling each feature against over all time in observation
        featdf=pd.DataFrame()
        
        status=row/dataset.shape[0]
        sys.stdout.write("\r Progress....{}".format(round(status,2)))
        sys.stdout.flush()
            
        for feat in features:
            #select all the features with partial match
            vec=d.columns.to_series().str.contains(feat)
            vec=d.columns[vec]
            d1=d[vec]
            # reshape for minmaxscaler (needs to shape off columns)
            # then scale the feature
            shaped=d1.values.reshape(d1.shape[-1],1)
            #scal = scaler.fit_transform(shaped)
            #scal = scal.reshape(1,scal.shape[0])
            #scal = pd.DataFrame(scal, columns = vec)
            # other method...
            if shaped[0]:
                pO=1
            else:
                pO=shaped[0]
                
            scal = [((float(p) / float(pO)) - 1) for p in shaped]
            featdf=pd.concat([featdf, scal], axis=1)
            
        scaled=scaled.append(featdf, ignore_index=True)
        
    return scaled


def shape(dataset, n_in, features):
    '''Shape data for LSTM input '''
    shaped = dataset.reshape(dataset.shape[0], n_in, len(features))
    return shaped

def unshape(dataset):
    '''Convert from LSTM to 2 dimensional '''
    dim2 = dataset.shape[2] * dataset.shape[1]
    unshaped = dataset.reshape((dataset.shape[0], dim2))
    return unshaped


# convert series to supervised learning
def series_to_supervised(data, features, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('{}(t-%d)'.format(features[j]) % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('{}(t)'.format(features[j])) for j in range(n_vars)]
        else:
            names += [('{}(t+%d)'.format(features[j]) % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def single_batch(data_adj,pred_len=1):
    # uses all prior data to predict all future data points
    # only select predict[0], first prediction. throw out the rest
    train_set_length = data_adj.shape[0]-(2*pred_len)+1
    train_set_width = data_adj.shape[1]

    train_X = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)
    train_Y = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)

    for i in range(0, train_set_length):
        start = i
        end =   i + pred_len
        train_X[i] = data_adj.ix[start:end].as_matrix()
        train_Y[i] = data_adj.ix[start+pred_len:end+pred_len].as_matrix()

    return train_X,train_Y

def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        # go to first row[index], select that row through -> sequence length 
        result.append(data[index: index + sequence_length])
        # assign that selection to new array, all in the same row
        # move on to the next row in data, repeat
    return np.asarray(result)


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

