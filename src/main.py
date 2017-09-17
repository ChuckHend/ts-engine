from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
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
n_in = 10
n_out = 1

# load dataset
# dataset = read_csv('stock_dfs/AMD.csv', header=0, index_col=0)
dataset = getStocks.get_single(ticker=ticker)

## Generate new features
dataset = fe.derivative(dataset, fill_na = True)
features = dataset.columns

# Plot the features
# visualize.plot_features(dataset)

# normalize features
# consider when to apply scaling
# before or after series_to_supervised?
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# frame as supervised learning
reframed = ps.series_to_supervised(scaled, n_in=n_in, n_out=n_out, 
                                   features=features)


# drop columns we don't want to predict
# drop t+1 features other than outcome variable
reframed.drop(['Open(t)', 'High(t)', 'Low(t)', 'Close(t)', 'Volume(t)',
               'd1close(t)', 'd2close(t)', 'd1vol(t)', 'd2vol(t)'],
    axis = 1, inplace=True)
 
# split into train, validation, test
values = reframed.values
train, test, validation = lstm.tscv(values)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


# reshape input to be 3D [samples, timesteps, features]
train_X = ps.shape(train_X, n_in=n_in, features=features)
test_X = ps.shape(test_X, n_in=n_in, features=features)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 

model = lstm.build_model(train_X, 
                         timesteps=n_in, 
                         inlayer=30,
                         hidden1=10, 
                         outLayer=1)

# fit network
# low epocs for testing/debug
history = model.fit(train_X, train_y, epochs=4, 
                    batch_size=100, validation_data=(test_X, test_y), 
                    verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
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
visualize.plot_single(predicted=yhat, actual=test_y, ticker=ticker)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(rmse)


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