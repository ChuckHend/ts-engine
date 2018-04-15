import featureEng as fe
import processStocks as ps
import visualize as vz
import pandas as pd
import time
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def build_model(dataObj, batch_size=None, inlayer=20, hiddenlayers=0, dropout=0.3,
                loss_function='mean_absolute_percentage_error',
                activation='tanh'):
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
    model.compile(loss=loss_function, optimizer="adam")
    print("Compilation Time : ", time.time() - start)
    model.summary()

    return model

class ts_data():

	def __init__(self, ticker, target, n_in=5, n_out=5, rawData=None):
		self.data = rawData
		self.n_in = n_in
		self.n_out = n_out
		self.ticker = ticker
		self.target = target

	@classmethod
	def default_prep(class_object, rawData, ticker, target, n_in=5, n_out=5):
		obj = class_object(rawData=rawData, ticker=ticker, target=target, n_in=n_in, n_out=n_out)
		#obj.eng_features()
		obj.roll_data()
		obj.tscv()
		return obj

	def eng_features(self,derivate=True, weekdays=True):
		if derivate:
			self.data = fe.derivative(self.data, drop_na=True)

		if weekdays:
			self.data = fe.weekDay(self.data)
		else:
			self.data.drop('Date', axis=1, inplace=True)
            

		features = list(self.data.columns)

		# move the target feature to position [-1] in dataframe
		features.remove(self.target)
		features.append(self.target)
		self.features = list(features)
		self.data = self.data[self.features]

	def roll_data(self):
		reframed = ps.series_to_supervised(self.data,
										   features=self.features,
										   n_in=self.n_in,
										   n_out=self.n_out)

		reframed = ps.frame_targets(reframed,
									features=self.features,
									n_out=self.n_out,
									target=self.target)

		self.data = reframed

	def tscv(self,train=0.95):
		# tscv - time series cross validation
		rows = self.data.shape[0]
		traincut = int(rows*train)

		train = self.data.values[:traincut, :]
		test = self.data.values[traincut:, :]

		self.train_X = ps.tensor_shape(train[:, :-self.n_out], self.n_in, self.features)
		self.train_y = train[:, -self.n_out:]

		self.test_X = ps.tensor_shape(test[:, :-self.n_out], self.n_in, self.features)
		self.test_y = test[:, -self.n_out:]
