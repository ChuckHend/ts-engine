import pandas as pd
import os, sys
#os.chdir('c:/Users/hende/onedrive/analytics/finance/4cast/src')
os.chdir('/Users/ahendel1/documents/academics/4cast/src')
import featureEng as fe
import processStocks as ps
import lstm
import getStocks
import visualize

####TODO: reshaping so we can plot various n_in, n_out
# model seems to work, but cant redim for plot
ticker = 'astc'

scaled_in = 6
scaled_out = 6
n_in = 5
n_out = 1
# load dataset
#dataset = getStocks.get_single(ticker=ticker, save=True)
dataset = getStocks.load_single(ticker)
dataset.rename(columns={'Adj Close':'AdjCls'}, inplace=True)

## Generate new features
dataset = fe.derivative(dataset, fill_na = True)

features = dataset.columns

# Plot the features
# visualize.plot_features(dataset)

# normalize or scale features
''' scaling might not even be necessary....'''

''' steps:
    # d = series_to_supervised for scaled_in and scaled_out
    # scale d for each feature, scaled_in to scaled_out
    # d = filter d for n_in and n_out (n_in < scaled_in, n_out < scaled_out)'''
# scaled = scaler.fit_transform(dataset)

# frame as supervised learning
# this will be for scaling the data to the window scaled_in to scaled_out
reframed = ps.series_to_supervised(dataset, n_in=scaled_in, n_out=scaled_out, 
                                   features=features)

#TODO: create dictionary to unscale each row of data
# note, can also keep a dictionary that maintains unscaling information
# iterate over every row
scaled=ps.scale_sequence(reframed, features)

# filter dataset down to n_in and n_out
featStrings=ps.get_filter_seq(features, n_in, n_out)        
scaled=scaled[featStrings]

# drop all but the 'target' from the predictor set
# this might be able to take an array for multi-output
scaled=ps.drop_targets(scaled, features, n_out,target='Close')


# split into train, validation, test
values = scaled.values
train, validation, test = lstm.tscv(values)

# split into input and outputs
# the last n columns are the output variable
train_X, train_y = train[:, :-n_out], train[:, -n_out:]
X_validation, Y_validation = validation[:, :-n_out], validation[:, -n_out:]
test_X, test_y = test[:, :-n_out], test[:, -n_out:]


# reshape input to be 3D [samples(observations), timesteps, features]
train_X = ps.shape(train_X, n_in=n_in, features=features)
X_validation = ps.shape(X_validation, n_in=n_in, features=features)
test_X = ps.shape(test_X, n_in=n_in, features=features)

print(train_X.shape, train_y.shape, 
      X_validation.shape, Y_validation.shape,
      test_X.shape, test_y.shape)
 

model = lstm.build_model(train_X, 
                         timesteps=n_in, 
                         inlayer=int(train_X.shape[-1]*15),
                         hiddenlayers=[256], 
                         outlayer=n_out)

# fit network and save to history
# low epocs for testing/debug
history = model.fit(train_X, train_y, epochs=10, 
                    batch_size=200, 
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


visualize.plot_single(predicted=scaler.inverse_transform(yhat), 
                      actual=scaler.inverse_transform(test_y), ticker=ticker)


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