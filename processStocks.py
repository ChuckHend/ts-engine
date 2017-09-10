import numpy as np
import matplotlib.pyplot as plt

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

def z_score(data):
    dates = data['Date']
    # z-score method. Normalizes by 'column' of data frame.
    data = data.apply(lambda x: (x - np.mean(x)) / np.std(x))
    for i in range(0,data.index.shape[0]):
        # take first row, convert date to ordinal form
        data_adj.loc[data.index[i],'Ordinal/1e6'] = data.index[i].to_pydatetime().toordinal()/1e6
        # and add weekday field
        data_adj.loc[data.index[i],'Weekday']     = data.index[i].to_pydatetime().weekday()
    
    return data


def stock_plot(data):
    # convert to tuple for plotting
    data = (data,)

    ax0 = plt.subplot2grid((6,2),(0,0),rowspan=5, colspan=1)
    ax1 = plt.subplot2grid((6,2),(5,0),rowspan=1, colspan=1, sharex=ax0)
    ax2 = plt.subplot2grid((6,2),(0,1),rowspan=5, colspan=1)
    ax3 = plt.subplot2grid((6,2),(5,1),rowspan=1, colspan=1, sharex=ax2)

    for each in data:
        ax0.plot(each.index,each['Adj Close'])
        ax1.plot(each.index,each['Adj Volume'])
        ax2.plot(each.index,each['Normalised Close'])
        ax3.plot(each.index,each['Normalised Volume'])

    plt.show()

def single_batch(data_adj,pred_len=1):
    # uses all prior data to predict all future data points
    # only select predict[0], first prediction. throw out the rest
    train_set_length = data_adj.shape[0]-(2*pred_len)+1
    train_set_width = data_adj.shape[1]

    train_X = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)
    train_Y = np.empty([train_set_length,pred_len,train_set_width],dtype=data_adj.dtypes)

    for i in range(0, train_set_length):
        start = i
        end =   i + pred_len
        train_X[i] = data_adj.ix[start:end].as_matrix()
        train_Y[i] = data_adj.ix[start+pred_len:end+pred_len].as_matrix()

    return train_X,train_Y

def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        # go to first row[index], select that row through -> sequence length 
        result.append(data[index: index + sequence_length])
        # assign that selection to new array, all in the same row
        # move on to the next row in data, repeat
    return np.asarray(result)
