import time
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential



def tscv(dataset,train=0.6, validation=0.2):
    # 'test' set is the remainder of data after train and validation
    # time series cross validation tscv
    rows = dataset.shape[0]
    traincut = int(rows*train)
    validationcut = int(rows*(train+validation))
    
    train = dataset.values[:traincut,:]
    validation = dataset.values[traincut:validationcut,:]
    test = dataset.values[validationcut:,:]
    return train, validation, test


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def build_model(x_train, timesteps, inlayer, outlayer,
                batch_size=None, hiddenlayers=0, dropout=0.3, loss_function='mean_absolute_percentage_error'):
    # outlayer is the number of predictions, days to predict
    # to run through before updating the weights
    # timesteps is the length of times, or length of the sequences
    # in each batch, input_dim is the number of features in each observation)
    input_dim = x_train.shape[-1]
    model = Sequential()
    # input layer
    if hiddenlayers==0:
        l1_seq=False
    else:
        l1_seq=True
        
    model.add(LSTM(
    #3D tensor with shape (batch_size, timesteps, input_dim)
    # (Optional) 2D tensors with shape  (batch_size, output_dim).
        #input_shape=(layers[1], layers[0]),
        input_shape=(timesteps, input_dim),
        units = inlayer,
        # output_dim=batch_size, #this might be wrong or need to be variable
        return_sequences=l1_seq,
        activation='tanh'
        ))
    model.add(Dropout(dropout))    
    
    #true by default
    seq=True
    if hiddenlayers!=0:
        for y, layer in enumerate(hiddenlayers):
            lastlayr=len(hiddenlayers)-1
            if y==lastlayr:
                seq=False
            model.add(LSTM(
                    units=layer,
                    return_sequences=seq,
                    activation='tanh'))
            model.add(Dropout(dropout))

    # output node   
    model.add(Dense(
        units=outlayer,
        activation='tanh'))

    start = time.time()
    model.compile(loss=loss_function, optimizer="adam")
    print("Compilation Time : ", time.time() - start)
    model.summary()

    return model

