from keras.models import load_model
from ts_data.ts_data import ts_data as ts
import pandas as pd
import sys
# n_in=10
# n_out=10
# ticker='AMD'
# n_pred=6

def validateInput():
    if len(sys.argv) != 5:
        print('[ERROR] USAGE: python predict.py ticker n_in n_out n_pred')
        sys.exit()
    else:
        return(
            str(sys.argv[1]), #ticker
            int(sys.argv[2]), #n_in
            int(sys.argv[3]), #n_out
            int(sys.argv[4])) #n_pred

def ts_transform(n_in, n_out, entityID, target, rawData):
    ts_data = ts(n_in=n_in, 
                 n_out=n_out, 
                 entityID=entityID,
                 target=target,
                 rawData=rawData)

    ts_data.eng_features(derivate=False, weekdays=False)

    ts_data.roll_data(train=False)

    # use train = 1 for live predictions
    # We should only be getting "X" data when we are predicting 
    # into unknown data points
    ts_data.tensor_shape()

    return ts_data

def main():
    ticker, n_in, n_out, n_pred = validateInput()
    
    # load model
    model = load_model('../models/lstm_{}_{}_{}.h5'.format(n_in, n_out, n_pred))

    # load new single record of data
    df = pd.read_csv('../../data/{}_plus_{}.csv'.format(ticker, n_pred))

    target='close_{}'.format(ticker)

    ts_data = ts_transform(
        n_in=n_in, 
        n_out=n_out, 
        entityID=ticker, 
        target=target, 
        rawData=df)

    # select the last sequence of data (it is length of n_in)
    test_X = ts_data.test_X[-1]

    # predict w/ new data
    yhat = model.predict(test_X.reshape(1, test_X.shape[-2], test_X.shape[-1]))

    print('\nNext {} prices for {}\n'.format(n_out,ticker))
    print(yhat[0])

    sys.exit()

if __name__ == "__main__":
    main()