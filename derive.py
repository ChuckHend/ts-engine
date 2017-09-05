# Feature Engineering
# Calculate d/dx and d2/dx2 for both close and Volume

def derivative(df):
    df['d1close'] = df.Close.diff()
    df['d2close'] = d1close.diff()
    df['d1vol'] = df.Volume.diff()
    df['d2vol'] = d1vol.diff()
    df = df.fillna(0)
    return df
