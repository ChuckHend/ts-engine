import ts_data.featureEng as fe
import ts_data.preprocess as ps
import visualize as vz
import pandas as pd
import time
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model

def lstm_model(dataObj, batch_size=None, inlayer=20, hiddenlayers=0, dropout=0.3,
                loss_function='mean_absolute_percentage_error',
                activation='tanh',gpus=1):
    # outlayer is the number of predictions, days to predict
    # to run through before updating the weights
    # timesteps is the length of times, or length of the sequences
    # in each batch, input_dim is the number of features in each observation)
    input_dim = dataObj.train_X.shape[-1]

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
        input_shape=(dataObj.n_in, input_dim),
        units = inlayer,
        # output_dim=batch_size, #this might be wrong or need to be variable
        return_sequences=l1_seq,
        activation=activation
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
                    activation=activation))
            model.add(Dropout(dropout))

    # output node
    model.add(Dense(
        units=dataObj.n_out,
        activation=None))

    start = time.time()
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)

    model.compile(loss=loss_function, optimizer="adam")
    print("Compilation Time : ", time.time() - start)
    model.summary()

    return model
