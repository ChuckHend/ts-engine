from ts_data import featureEng as fe
from ts_data import preprocess as ps


class ts_data():

    def __init__(self, target, n_in=5, n_out=5, entityID=None, rawData=None):
        self.data = rawData
        self.n_in = n_in
        self.n_out = n_out
        self.entityID = entityID
        self.target = target
        self.features = list(rawData.columns)

    @classmethod
    def default_prep(class_object, rawData, entityID, target, n_in=5, n_out=5):
        obj = class_object(rawData=rawData, entityID=entityID, target=target, n_in=n_in, n_out=n_out)
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
        print('Processing: series_to_supervised()')
        reframed = ps.series_to_supervised(self.data,
                                           features=self.features,
                                           n_in=self.n_in,
                                           n_out=self.n_out)
        print('Processing: frame_targets()')
        reframed = ps.frame_targets(reframed,
                                    features=self.features,
                                    n_out=self.n_out,
                                    target=self.target)
        print('Total Supervised Learning Records: {}'.format(reframed.shape[0]))

        self.data = reframed

    def tscv(self,train=0.95):
        # tscv - time series cross validation
        rows = self.data.shape[0]
        traincut = int(rows*train)

        train = self.data.values[:traincut, :]
        test = self.data.values[traincut:, :]
        # the outcome variable (y) will be in position [,-n_out:]
        # i.e. the outcome variable is on right end of the matrix
        self.train_X = ps.tensor_shape(train[:, :-self.n_out], self.n_in, self.features)
        self.train_y = train[:, -self.n_out:]

        self.test_X = ps.tensor_shape(test[:, :-self.n_out], self.n_in, self.features)
        self.test_y = test[:, -self.n_out:]