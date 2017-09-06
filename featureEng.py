# Feature Engineering
# Calculate d/dx and d2/dx2 for both close and Volume

def derivative(df, fill_na = True):
    df['d1close'] = df.Close.diff()
    df['d2close'] = df.Close.diff().diff()
    df['d1vol'] = df.Volume.diff()
    df['d2vol'] = df.Volume.diff().diff()
    if fill_na:
        df = df.fillna(0)
    return df
