# Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# create day of week from date
def weekDay(dataset, drop_na=True):
    #Monday=0, Sunday=6
    date=pd.to_datetime(dataset['Date'])
    dataset.drop('Date', axis=1, inplace=True)
    dow=date.dt.dayofweek
    enc = OneHotEncoder()
    dat=enc.fit_transform(dow.values.reshape(-1,1)).toarray()
    dat=pd.DataFrame(dat, columns=['M','T','W','Th','F'])
    dat=dat.reset_index(drop=True)
    dataset=dataset.reset_index(drop=True)
    dataset=pd.concat([dataset, dat], axis=1)
    if drop_na:
        dataset=dataset.dropna(0)
    return dataset

# Calculate d/dx and d2/dx2 for both close and Volume
def derivative(df, drop_na = True):
    df['d1close'] = df.Close.diff()
    df['d2close'] = df.Close.diff().diff()
    df['d1vol'] = df.Volume.diff()
    df['d2vol'] = df.Volume.diff().diff()
    if drop_na:
        df = df.dropna(0)
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
