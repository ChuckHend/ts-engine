import featureEng as fe
import processStocks as ps
import visualize as vz
import pandas as pd
import time
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class TS_LSTM():

	def __init__(self, ticker, target, n_in=5, n_out=5, rawData=None):
		self.data = rawData
		self.n_in = n_in
		self.n_out = n_out
		self.ticker = ticker
		self.target = target

	def eng_features(self,derivate=True, weekdays=True):
		if derivate:
			self.data = fe.derivative(self.data, drop_na=True)

		if weekdays:
			self.data = fe.weekDay(self.data)

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

	def train_model(self, epochs=50, batch_size=256, plot=True):
		self.history = self.model.fit(self.train_X, self.train_y,
			epochs=epochs, validation_data=(self.test_X, self.test_y),
			verbose=2, shuffle=False)

		if plot:
			vz.plot_loss(self.history)



	def build_model(self, batch_size=None, hiddenlayers=0, dropout=0.3, 
	                loss_function='mean_absolute_percentage_error'):
	    # outlayer is the number of predictions, days to predict
	    # to run through before updating the weights
	    # timesteps is the length of times, or length of the sequences
	    # in each batch, input_dim is the number of features in each observation)
	    input_dim = self.train_X.shape[-1]

	    inlayer = int(self.train_X.shape[-1]*.75)

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
	        input_shape=(self.n_in, input_dim),
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
	        units=self.n_out,
	        activation='tanh'))

	    start = time.time()
	    model.compile(loss=loss_function, optimizer="adam")
	    print("Compilation Time : ", time.time() - start)
	    model.summary()

	    self.model = model

	def tscv(self,train=0.8):
		# tscv - time series cross validation
		rows = self.data.shape[0]
		traincut = int(rows*train)

		train = self.data.values[:traincut, :]
		test = self.data.values[traincut:, :]

		self.train_X = ps.tensor_shape(train[:, :-self.n_out], self.n_in, self.features)
		self.train_y = train[:, -self.n_out:]

		self.test_X = ps.tensor_shape(test[:, :-self.n_out], self.n_in, self.features)
		self.test_y = test[:, -self.n_out:]



