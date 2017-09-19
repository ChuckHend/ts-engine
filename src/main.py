from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
import featureEng as fe
import processStocks as ps
import lstm
import getStocks
import visualize
from sklearn.preprocessing import MinMaxScaler

####TODO: reshaping so we can plot various n_in, n_out
# model seems to work, but cant redim for plot
ticker = 'unh'
n_in = 1
n_out = 1

# load dataset
# dataset = read_csv('stock_dfs/AMD.csv', header=0, index_col=0)
dataset = getStocks.get_single(ticker=ticker)

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
 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# frame as supervised learning
reframed = ps.series_to_supervised(scaled, n_in=n_in, n_out=n_out, 
                                   features=features)

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
                         inlayer=int(train_X.shape[-1]*1.5),
                         hiddenlayers=[550], 
                         outlayer=n_out)

# fit network and save to history
# low epocs for testing/debug
history = model.fit(train_X, train_y, epochs=6, 
                    batch_size=100, 
                    validation_data=(X_validation, Y_validation), 
                    verbose=2, shuffle=False)
# plot history
visualize.plot_loss(history)

# make a prediction
yhat = model.predict(test_X)
# convert back to 2d
test_X = ps.unshape(test_X)

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# plot unscaled
visualize.plot_single(predicted=inv_yhat, actual=inv_y, ticker=ticker)
# plot scaled
#TODO: viz method. to plot the sequences...


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