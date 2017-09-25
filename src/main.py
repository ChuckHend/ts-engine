from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
os.chdir('c:/Users/hende/onedrive/analytics/finance/4cast/src')
import featureEng as fe
import processStocks as ps
import lstm
import getStocks
import visualize
from sklearn.preprocessing import MinMaxScaler

####TODO: reshaping so we can plot various n_in, n_out
# model seems to work, but cant redim for plot
ticker = 'astc'
n_in = 1
n_out = 1

# load dataset
# dataset = getStocks.get_single(ticker=ticker, save=True)
dataset = getStocks.load_single(ticker)
dataset.rename(columns={'Adj Close':'AdjCls'}, inplace=True)

## Generate new features
dataset = fe.derivative(dataset, fill_na = True)

features = dataset.columns

# Plot the features
# visualize.plot_features(dataset)

# normalize or scale features
''' consider when to apply scaling//before or after series_to_supervised?
timeframe is important. do we want to scale based on entire dataset, or just
based on the sequence length? for example, if we are predicting the next day's
price based on the last 10 days activity, should the activity and target be scaled
to the past 10 days, or to the entire dataset? or to something else? I would lean
towards scaling to the sequence length (ie past 10 days) and include features that
will have information related to the overall period'''
'''could also convert simply to stationary'''

''' should select a range to look back for scaling and select a different 
    range for the n_in and n_out...something like -30 or 1 year for scaling, then
    select n_in and n_out from the scaled dataset'''
''' steps:
    # d = series_to_supervised for scaled_in and scaled_out
      d = filter d for n_in and n_out (n_in < scaled_in, n_out < scaled_out)'''
# scaled = scaler.fit_transform(dataset)

# frame as supervised learning
reframed = ps.series_to_supervised(dataset, n_in=n_in, n_out=n_out, 
                                   features=features)


#TODO: package to function
# note, can also keep a dictionary that maintains unscaling information
# iterate over every row
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled=pd.DataFrame(columns=reframed.columns)
for row in range(reframed.shape[0]):
    d=reframed.iloc[row,:]
    d=pd.DataFrame(d.values.reshape(1,d.shape[0]), columns=reframed.columns)
    # iterate over each feature
    # scaling each feature against over all time in observation
    featdf=pd.DataFrame()
    print('Processing row: {}....{} complete'.format(row, row/reframed.shape[0]))
    for feat in features:
        #select all the features with partial match
        vec=d.columns.to_series().str.contains(feat)
        vec=d.columns[vec]
        d1=d[vec]
        # reshape for minmaxscaler (needs to shape off columns)
        # then scale the feature
        shaped=d1.values.reshape(d1.shape[-1],1)
        scal = scaler.fit_transform(shaped)
        scal = scal.reshape(1,scal.shape[0])
        scal = pd.DataFrame(scal, columns = vec)
        featdf=pd.concat([featdf, scal], axis=1)
        
    #print(featdf)
    scaled=scaled.append(featdf, ignore_index=True)
###END TODO:

# create the list of features we dont want to predict
# so take all features, less the one we want to predict
dropList = list(features)
dropList.remove('Close')

# iterate over droplist, removing all columns in the droplist
# for t....t+n...which will be t+n_in-1
for feature in dropList:
    # iterate over n_out-1
    reframed.drop('{}(t)'.format(feature), axis=1, inplace=True)
    #print('dropped {}(t)'.format(feature))
    for i in range(1,n_out):
        tfeat='{}(t+{})'.format(feature,i)
        reframed.drop(tfeat, axis=1, inplace=True)
        #print('dropped {}'.format(tfeat))


# split into train, validation, test
values = reframed.values
train, validation, test = lstm.tscv(values)

# split into input and outputs
# the last n columns are the output variable
train_X, train_y = train[:, :-n_out], train[:, -n_out:]
X_validation, Y_validation = validation[:, :-n_out], validation[:, -n_out:]
test_X, test_y = test[:, :-n_out], test[:, -n_out:]


# reshape input to be 3D [samples, timesteps, features]
train_X = ps.shape(train_X, n_in=n_in, features=features)
X_validation = ps.shape(X_validation, n_in=n_in, features=features)
test_X = ps.shape(test_X, n_in=n_in, features=features)

print(train_X.shape, train_y.shape, 
      X_validation.shape, Y_validation.shape,
      test_X.shape, test_y.shape)
 

model = lstm.build_model(train_X, 
                         timesteps=n_in, 
                         inlayer=int(train_X.shape[-1]*30),
                         hiddenlayers=[300,300,300], 
                         outlayer=n_out)

# fit network and save to history
# low epocs for testing/debug
history = model.fit(train_X, train_y, epochs=50, 
                    batch_size=100, 
                    validation_data=(X_validation, Y_validation), 
                    verbose=2, shuffle=False)
# plot history
visualize.plot_loss(history)

# make a prediction
yhat = model.predict(test_X)
# convert back to 2d
#test_X = ps.unshape(test_X)

# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:,0]

# invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,0]


visualize.plot_single(predicted=yhat, actual=test_y, ticker=ticker)



###### TODO METHOD####
#import predicts
## copy weights
#old_weights=model.get_weights()
#
#new_model = Sequential()
#new_model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),
#          return_sequences=True, activation='tanh'))
#new_model.add(Dropout(0.5))
#new_model.add(LSTM(10, activation='tanh'))
#new_model.add(Dropout(0.5))
#new_model.add(Dense(1))
#new_model.compile(loss='mae', optimizer='adam')
## fit network
#new_model.set_weights(old_weights)
#predicts.onlineForecast(new_model, test_X, test_y)
## # # ____ # # #