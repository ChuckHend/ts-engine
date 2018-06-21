# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import sys
import numpy as np
import pandas as pd


def frame_targets(dataset, features, n_out,target='Close'):
    '''create the list of features we dont want to predict
    so take all features, less the one we want to predict
    example: if we are trying to predict a stock's closing price, two days into the future,
    Close(t+1) and Close(t+2), we'll want to remove the variables labels
    Volume(t+1), Volume(t+2), etc.
    TODO: should this be done in timeseries_to_supervised() ?
    '''
    dropList = list(features)
    dropList.remove(target)

    # iterate over droplist, removing all columns in the droplist
    # for t....t+n...which will be t+n_in-1
    for feature in dropList:
        # iterate over n_out-1
        dataset.drop('{}(t)'.format(feature), axis=1, inplace=True)

        # drop all t+ that arent our target feature
        for i in range(1,n_out):
            tfeat='{}(t+{:02})'.format(feature,i)
            dataset.drop(tfeat, axis=1, inplace=True)

    return dataset

def get_filter_seq(features, n_in, n_out):
    # creates the vector of columns we want from the outer scaled dataset
    # 'outer' is a set of t-n to t+n
    featStrings=[]
    for feat in features:
        featStrings.append('{}(t)'.format(feat))

        for x_in in range(1,n_in+1):
            featStrings.append('{}(t-{:02})'.format(feat, x_in))

        for x_out in range(1,n_out): # t+1 would be n_out=2
            featStrings.append('{}(t+{:02})'.format(feat, x_out))
    return featStrings


def find(lst, a): # return index of match
    return [i for i, x in enumerate(lst) if x==a]


def scale_sequence(dataset, features, scaleTarget=True, target='Close'):
    # accepts a reframed dataset (already in supervised time series)
    # scaled each observation, based on the data in the observation
    # on a feature by feature basis
    # features are the global feature set(prior to time series conv.)
    # use normalise window form instead of monmax

    # seqfeats are the features in sequence form
    #NOTE: dataset.columns ARE in the correct time sequence!

    ### DO A HARD COPY we want the order of the columns going out to be the same
    # order that they were coming in
    out_order=dataset.columns[:]
    seqfeats=dataset.columns

    # do not scale the weekdays (they are already one-hot encoded)
    days=['M','T','W','Th','F']
    vec= [False] * len(dataset.columns)
    for day in days:
        vec_i=seqfeats.to_series().str.contains(day)
        vec=np.logical_or(vec, vec_i)

    excl_cols=list(seqfeats[vec])

    # build list of columns to exclude from scaling
    # save the target features for positioning later
    tO='{}(t)'.format(target)
    tO=find(seqfeats, tO)
    tgt_seq=seqfeats[tO[0]:]
    if(not scaleTarget):
        excl_cols.extend(list(tgt_seq))

    seqfeats=seqfeats.drop(excl_cols)
    features=[x for x in features if x not in days]

    # dataset=what we will be scaling by sequence

    dow=dataset[excl_cols] # the data that isnt being scaled

    #then drop the days of week from the working dataset
    dataset=dataset.drop(excl_cols, axis=1) # data to be scaled

    # create a dictionary for the variables we'll be iterating over
    colDict={}
    for feature in features:
        # which columns contain partial match of this feature?
        cols=dataset.columns.to_series().str.contains(feature)
        cols=dataset.columns[cols]
        colDict[feature]=cols

    scaled=pd.DataFrame(columns=dataset.columns)
    for row in range(dataset.shape[0]): # row by row scaling each feature sequence
        print(row)
        print(dataset.shape)
        d=dataset.iloc[row,:]
        d=pd.DataFrame(d.values.reshape(1,d.shape[0]),
                       columns=dataset.columns)
        # iterate over each feature
        # scaling each feature against over all time in observation

        # print the current status to console
        status=row/dataset.shape[0]
        sys.stdout.write("\r Progress....{}".format(round(status,2)))
        sys.stdout.flush()

        featdf=pd.DataFrame()
        for feature in colDict: # level each feature in the row-sequence
            #select all the features with partial match

            vec=colDict[feature]
            d1=d[vec]
            # reshape for minmaxscaler (needs to shape off columns)
            # then scale the feature
            shaped=d1.values.reshape(d1.shape[-1],1)

            if shaped[0]==0:
                pO=1
            else:
                pO=float(shaped[0])

            scal = [((float(p) / pO) - 1) for p in shaped]
            scal=pd.DataFrame(scal).T
            scal.columns=vec
            featdf=pd.concat([featdf, scal], axis=1)

        scaled=scaled.append(featdf) # append the next row

    scaled=scaled.reset_index(drop=True)
    dow=dow.reset_index(drop=True)

    # need to make sure target variables get on the 'far right'
    if scaleTarget:
        # now move the target features to the right
        tgt=scaled[tgt_seq]
        scaled=scaled.drop(list((tgt_seq)),axis=1)
        scaled=pd.concat([dow, scaled, tgt], axis=1)
    else:
        scaled=pd.concat([scaled, dow], axis=1)

    scaled=scaled[out_order]
    print('Sequence scaling complete.')
    return scaled


def tensor_shape(dataset, n_in, features):
    #Shape data for LSTM input
    '''tensor should be (t-2)a, (t-2)b, (t-1)a, (t-1)b, etc.
     where a and b are features to properly reshape'''
    shaped = dataset.reshape(dataset.shape[0], n_in, len(features))
    return shaped

def unshape(dataset):
    #Convert from LSTM to 2 dimensional

    dim2 = dataset.shape[2] * dataset.shape[1]
    unshaped = dataset.reshape((dataset.shape[0], dim2))
    return unshaped


# convert series to supervised
def series_to_supervised(data, features, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    print('Processing input sequences')    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('{}(t-%02d)'.format(features[j]) % (i)) for j in range(n_vars)]
        status = round(i/n_in,2)
        sys.stdout.write("\r Progress....{}".format(round(status,2)))
        sys.stdout.flush()
    # forecast sequence (t, t+1, ... t+n)
    print('Processing output sequences') 
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('{}(t)'.format(features[j])) for j in range(n_vars)]
        else:
            names += [('{}(t+%02d)'.format(features[j]) % (i)) for j in range(n_vars)]
        status = round(i/n_out,2)
        sys.stdout.write("\r Progress....{}".format(round(status,2)))
        sys.stdout.flush()        
    # put it all together
    agg = pd.concat(cols, axis=1)
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
