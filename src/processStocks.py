import numpy as np
from pandas import DataFrame
from pandas import concat


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
