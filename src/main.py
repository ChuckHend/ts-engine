import os
#os.chdir('/Users/ahendel1/documents/academics/4cast/src')
os.chdir('D:/4cast/src')
import featureEng as fe
import processStocks as ps
import lstm
import getStocks
import visualize
import predicts

####TODO: reshaping so we can plot various n_in, n_out
# model seems to work, but cant redim for plot
ticker = 'UNH'

n_in = 50
n_out = 30
target = 'd1close'

# load dataset
dataset = getStocks.join_tgt_spt(target_ticker=ticker, number_spt=25)
dataset.reset_index(level=0, inplace=True)
# dataset = getStocks.get_single(ticker=ticker, save=True)
# dataset = getStocks.load_single(ticker)
# dataset.rename(columns={'Adj Close':'AdjCls'}, inplace=True)

## Generate new features
dataset = fe.derivative(dataset, drop_na = True)
dataset = fe.weekDay(dataset)
features = list(dataset.columns)
# move target to position [-1]
features.remove(target)
features.append(target)
features=list(features)
dataset=dataset[features]
# Plot the features
# visualize.plot_features(dataset)

# normalize or scale features
''' scaling might not even be necessary....but it probably is'''

''' steps:
    # d = series_to_supervised for scaled_in and scaled_out
    # scale d for each feature, scaled_in to scaled_out
    # d = filter d for n_in and n_out (n_in < scaled_in, n_out < scaled_out)'''
    # perhaps this should be z-score scaling

# frame as supervised learning
# this will be for scaling the data to the window scaled_in to scaled_out
reframed=ps.series_to_supervised(dataset, n_in=n_in, n_out=n_out,
                                   features=features)

# drop all but the 'target' from the predictor set
# this might be able to take an array for multi-output
reframed=ps.frame_targets(reframed, features, n_out,target=target)

#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaled = scaler.fit_transform(reframed)
# pass the feature in the sequence that we dont want scaled

#reframed=reframed.iloc[:10,:] #test/debug
#dataset=reframed
scaled=ps.scale_sequence(reframed, features,
                         scaleTarget=True, target=target)

#getStocks.saveScaled(scaled, n_in, n_out, ticker)
# load scaled

#scaled=pd.read_csv('../data/astc/astc_29092017_scaled_30_30.csv')


# put in order of time (t-n_in should be on left of df, with t+ on the right)

# split into train, validation, test
train, validation, test = lstm.tscv(scaled, train=0.7, validation=0.2)

# split into input and outputs
# the last n columns are the output variable

train_X, train_y = train[:, :-n_out], train[:, -n_out:]
X_validation, Y_validation = validation[:, :-n_out], validation[:, -n_out:]
test_X, test_y = test[:, :-n_out], test[:, -n_out:]



# ensure the shaping makes sense the way all the variables are arranged
# reshape input to be 3D [samples(observations), timesteps (n_in), features]
train_X = ps.shape(train_X, n_in=n_in, features=features)
X_validation = ps.shape(X_validation, n_in=n_in, features=features)
test_X = ps.shape(test_X, n_in=n_in, features=features)


model = lstm.build_model(train_X,
                         timesteps=n_in,
                         inlayer=int(train_X.shape[-1]*10),
                         hiddenlayers=[100],
                         outlayer=n_out)
# fit network and save to history
history = model.fit(train_X, train_y,
                    epochs=50,
                    batch_size=100,
                    validation_data=(X_validation, Y_validation),
                    verbose=2, shuffle=False)

# plot history
visualize.plot_loss(history)


### KEEP THIS WORKING ####
yhat = predicts.predict_sequences_multiple(model, test_X, n_in, n_out)
visualize.plot_results_multiple(yhat, test_y[:,0], n_out, legend=False)
# visualize.plot_single(yhat, test_y, ticker, data_set_category='test')

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
