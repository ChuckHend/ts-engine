import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to disable GPU
from ts_data.ts_data import ts_data as ts
import ts_models.predicts
import pandas as pd
import ts_models.ts_lstm as ts_lstm
import numpy as np
import time, sys
from ts_config import load_config as cfg
import pickle
from keras.callbacks import EarlyStopping

def validateInput():
    if len(sys.argv) != 5:
        print('[ERROR] USAGE: python intra-day-prep.py <target-ticker> <n_in> <n_out> <n_pred> ')
        sys.exit()
    else:
        return(
          str(sys.argv[1]).upper(), #ticker
          int(sys.argv[2]),         #n_in
          int(sys.argv[3]),         #n_out
          int(sys.argv[4]))         #n_pred

def transform_data(n_in, n_out, entityID, target, dataset):
    ts_data = ts(n_in=n_in, 
                 n_out=n_out, 
                 entityID=entityID,
                 target=target,
                 rawData=dataset)

    ts_data.eng_features(derivate=False, weekdays=False)

    ts_data.roll_data()

    ts_data.tscv(train=0.98)

    return ts_data

def model_fit(ts_data):
    early_stopping = EarlyStopping(
      monitor='val_loss',
      min_delta=.05,
      patience=3,
      mode='min'
      )

    num_gpu = cfg('num_gpu')

    ts_model = ts_lstm.lstm_model(
      ts_data, 
      inlayer=int(ts_data.train_X.shape[-1])*2,
      hiddenlayers=[256],
      loss_function='mae',
      dropout=0.15,
      activation='tanh',
      gpus=num_gpu)

    start = time.time()

    debug = cfg('debug')
    if debug:
      epochs = 5
    else:
      epochs = 150
    history = ts_model.fit(
      ts_data.train_X, ts_data.train_y,
      epochs=epochs, 
      batch_size=1024, 
      validation_data=(ts_data.test_X, ts_data.test_y),
      verbose=2, 
      shuffle=False,
      callbacks=[early_stopping]
      )
    fitTime = time.time()-start
    print('Fit Time: {}'.format(round(fitTime,2)))

    return ts_model

def check_make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_model(model, n_in, n_out, n_predictors):
    model_folder = './models'
    # make the 'models' folder if it does not exist
    check_make_dir(model_folder)
    modpath = '{}/lstm_{}_{}_{}.h5'.format(
      model_folder,
      n_in, 
      n_out, 
      n_predictors)
    model.save(modpath)
    print('Model saved to:\n {}'.format(modpath))


def main():
    loadpkl=cfg('load_transformed')
    if loadpkl:
        print('Loading pickle')
        data_obj = pickle.load(open('src/ts_data/ts_data.pkl','rb'))

    else:
      target = '{}_{}'.format(
        cfg('outcome_variables'),
        cfg('target'))
      entity = cfg('target')
      n_pred = cfg('n_pred')
      data = pd.read_csv('../data/{}_plus_{}.csv'.format(entity, n_pred))
      data_obj = transform_data(
        n_in=cfg('n_in'), 
        n_out=cfg('n_out'), 
        entityID=cfg('target'), 
        target=target, 
        dataset=data)

    model = model_fit(data_obj)
    
    save_model(
      model=model, 
      n_in=cfg('n_in'), 
      n_out=cfg('n_out'), 
      n_predictors=cfg('n_pred'))

    AWS=False
    if AWS:
        print('Pickling ts_data.pkl')
        with open('ts_data.pkl', 'wb') as out:
            pickle.dump(data_obj,out,pickle.HIGHEST_PROTOCOL)
        sys.exit()

    sys.exit()

if __name__ == "__main__":
    main()
