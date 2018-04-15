# test dimension modification
import pandas as pd
df=pd.read_csv('testDim.csv')
features_test=list(df.columns)
df.head()


n_in=2
n_out=1
tgt_tst='Var4'
df=ps.series_to_supervised(df,
                           n_in=n_in,
                           n_out=n_out,
                           features=features_test)
df.head()


test_frame=ps.frame_targets(df,
                          features=features_test,
                          n_out=n_out,
                          target=tgt_tst)
test_frame.head()

X = test_frame.values[:, :-n_out]

[print(x) for x in X]


d=ps.shape(X, n_in, features_test)

d[1,-1,:]

'''
dim is (observations, n_in, features), which is correct

tensor should be (t-2)a, (t-2)b, (t-1)a, (t-1)b, etc. where a and b are
features to properly reshape, thus current reshaping is working as expected'''
