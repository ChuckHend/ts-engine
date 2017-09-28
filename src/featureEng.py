# Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# create day of week from date
def weekDay(dataset):
    #Monday=0, Sunday=6
    date=pd.to_datetime(dataset['Date'])
    dataset.drop(['Date'], axis=1, inplace=True)
    dow=date.dt.dayofweek
    enc = OneHotEncoder()
    dat=enc.fit_transform(dow.values.reshape(-1,1)).toarray()
    dat=pd.DataFrame(dat, columns=['M','T','W','Th','F'])
    dataset=pd.concat([dataset, dat], axis=1)
    
# Calculate d/dx and d2/dx2 for both close and Volume
def derivative(df, fill_na = True):
    df['d1close'] = df.Close.diff()
    df['d2close'] = df.Close.diff().diff()
    df['d1vol'] = df.Volume.diff()
    df['d2vol'] = df.Volume.diff().diff()
    if fill_na:
        df = df.fillna(0)
    return df

# integer encode string variables
def getDummies(df):
    df = pd.get_dummies(df)
    return df

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def z_score(data):
    data_adj = data
    # z-score method. Normalizes by 'column' of data frame.
    data = data.apply(lambda x: (x - np.mean(x)) / np.std(x))
    for i in range(0,data.index.shape[0]):
        # take first row, convert date to ordinal form
        data_adj.loc[data.index[i],'Ordinal/1e6'] = data.index[i].to_pydatetime().toordinal()/1e6
        # and add weekday field
        data_adj.loc[data.index[i],'Weekday']     = data.index[i].to_pydatetime().weekday()
    return data


def normalize_stock_data(data):
    ''' This function does not generalize well. It calls columns by name, so as 
    new columns (features) are added to the dataset, this will not work. As of 
    10-Sep-2017, already not working because I added derivative features'''
    
    data_adj=data

    for i in range(0,data.index.shape[0]):
        # take first row, convert date to ordinal form
        data_adj.loc[data.index[i],'Ordinal/1e6'] = data.index[i].to_pydatetime().toordinal()/1e6
        # and add weekday field
        data_adj.loc[data.index[i],'Weekday']     = data.index[i].to_pydatetime().weekday()

    # drop the non-normalized from new DF (ie select the normalized fields)
    # colNames = data.columns
    
    data_adj = data.drop(['Open','High','Low','Close','Adj Close', 'Volume'], axis = 1)

    # make 'Adj' columns by diving adj close by 'close'
    data_adj['Adj'] = data['Adj Close']/data['Close']

    # make an adj volume column
    data_adj['Adj Volume'] = data['Volume']
    # divide the entire column by the max value in the column
    data_adj['Adj Volume'] /= np.max(data_adj['Adj Volume'])

    data_adj['Adj Close'] = data['Adj Close'] / data['Adj Close'][0]
    data_adj['Adj Open'] = data['Open']*data_adj['Adj'] / data['Adj Close'][0]
    data_adj['Adj High'] = data['High']*data_adj['Adj'] / data['Adj Close'][0]
    data_adj['Adj Low']  = data['Low'] *data_adj['Adj'] / data['Adj Close'][0]

    data_adj.loc[data.index[0],'Normalised Volume'] = 1
    data_adj.loc[data.index[1:],'Normalised Volume'] = data_adj['Adj Volume'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Volume'] -= 1

    data_adj.loc[data.index[0],'Normalised Close'] = 1
    data_adj.loc[data.index[1:],'Normalised Close'] = data_adj['Adj Close'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Close'] -= 1

    data_adj.loc[data.index[0],'Normalised Open'] = 1
    data_adj.loc[data.index[1:],'Normalised Open'] = data_adj['Adj Open'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Open'] -= 1

    data_adj.loc[data.index[0],'Normalised High'] = 1
    data_adj.loc[data.index[1:],'Normalised High'] = data_adj['Adj High'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised High'] -= 1

    data_adj.loc[data.index[0],'Normalised Low'] = 1
    data_adj.loc[data.index[1:],'Normalised Low'] = data_adj['Adj Low'][1:] / data_adj['Adj Close'][:-1].values
    data_adj.loc[data.index,'Normalised Low'] -= 1

    #reduce some mean
    data_adj=data_adj.drop(['Adj'], axis=1)

    return data_adj