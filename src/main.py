from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import pandas_datareader
import datetime as dt
from datetime import timedelta
import featureEng as fe
from processStocks import series_to_supervised
import lstm

ticker = 'mmm'
n_in = 1
n_out = 1
start = dt.datetime(1995,1,1)
end   = dt.date.today()

# load dataset
# dataset = read_csv('stock_dfs/AMD.csv', header=0, index_col=0)
dataset = pandas_datareader.data.DataReader(ticker,'yahoo',start,end)
dataset = fe.derivative(dataset, fill_na = True)
features = dataset.columns
dataset.head()
values = dataset.values
groups = [0, 1, 2, 3, 5]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, n_in=n_in, n_out=n_out, features=features)
reframed.head()
# drop columns we don't want to predict
# ie drop time zero features other than the predictor
# reframed.drop(reframed.columns[[-1, -3, -5, -6, -7]], axis=1, inplace=True)
reframed.drop(['Open(t)', 'High(t)', 'Low(t)', 'Close(t)', 'Volume(t)',
               'd1close(t)', 'd2close(t)', 'd1vol(t)', 'd2vol(t)'],
    axis = 1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = int(values.shape[0]*.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_in, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], n_in, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
import imp
imp.reload(lstm)
# try to sub lstm function for model architecture
model = lstm.build_model(train_X, 
                         timesteps=n_in, 
                         inlayer=30,
                         hidden1=10, 
                         outLayer=1)

# fit network
history = model.fit(train_X, train_y, epochs=12, 
                    batch_size=100, validation_data=(test_X, test_y), 
                    verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
d = dt.datetime.today() - timedelta(days = len(test_X))
d = d.year
pyplot.title('{} Predicted Stock Daily Close (test)'.format(ticker.upper()))
pyplot.plot(inv_y, label='Actual Test')
pyplot.plot(inv_yhat, label='Predicted Test')
pyplot.ylabel('Daily Close')
pyplot.xlabel('{} to present '.format(d))
pyplot.legend()
pyplot.show()
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(rmse)




# # # TODO METHOD # # # 
# predict on training data
yhat_trn = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
# invert scaling for actual training
inv_yhat_trn = concatenate((yhat_trn, train_X[:, 1:]), axis=1)
inv_yhat_trn = scaler.inverse_transform(inv_yhat_trn)
inv_yhat_trn = inv_yhat_trn[:,0]

# invert tscaling for predicted training
train_y = train_y.reshape((len(train_y), 1))
inv_y_trn = concatenate((train_y, train_X[:, 1:]), axis=1)
inv_y_trn = scaler.inverse_transform(inv_y_trn)
inv_y_trn = inv_y_trn[:,0]

d = dt.datetime.today() - timedelta(days = len(test_X))
d = d.year
pyplot.title('{} Predicted Stock Daily Close (test)'.format(ticker.upper()))
pyplot.plot(inv_y, label='Actual Test')
pyplot.plot(inv_yhat, label='Predicted Test')
pyplot.ylabel('Daily Close')
pyplot.xlabel('{} to present '.format(d))
pyplot.legend()
pyplot.show()
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# plot training data
pyplot.title('{} Predicted Stock Daily Close (train)'.format(ticker.upper()))
pyplot.plot(inv_yhat_trn, label='Predicted Train')
pyplot.plot(inv_y_trn, label='Actual Train')
pyplot.ylabel('Daily Close')
pyplot.xlabel('{} to present '.format(d))
pyplot.legend()
pyplot.show()
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# # # ___ # # # 


##### TODO METHOD####
import predicts
# copy weights
old_weights=model.get_weights()

new_model = Sequential()
new_model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),
          return_sequences=True, activation='tanh'))
new_model.add(Dropout(0.5))
new_model.add(LSTM(10, activation='tanh'))
new_model.add(Dropout(0.5))
new_model.add(Dense(1))
new_model.compile(loss='mae', optimizer='adam')
# fit network
new_model.set_weights(old_weights)
predicts.onlineForecast(new_model, test_X, test_y)
# # # ____ # # #