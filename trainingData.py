import featureGenerator as fg
import pandas as pd
from pandas_datareader import data
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

class trainingData(object):

    index = 2 # number of datasets?
    #lags = range(2, 3)

    def load(self, fout, startdate, enddate, delta, lags):
        """
        Loads all data from web into dataFrames, 
        adds various features, merges datasets.
        """
        self.trainData = self.loadAllFromWeb(fout, startdate, enddate)
        self.trainDataWithFeatures = fg.applyRollMeanDelayedReturns(self.trainData, delta)
        self.mergedData = fg.mergeDataframes(self.trainDataWithFeatures, self.index, enddate)
        self.finalTrainingData = fg.applyTimeLag(self.mergedData, lags, delta)


    def getStockFromYahoo(self, symbol, start, end):
        """
        Downloads Stock from Yahoo Finance.
        Computes daily Returns based on Adj Close.
        Returns pandas dataframe.
        """
        df =  data.get_data_yahoo(symbol, start, end)

        df.columns.values[-1] = 'AdjClose'
        df.columns = df.columns + '_' + symbol
        df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    
        return df


    def getStockFromQuandl(self, symbol, name, start, end):
        """
        Downloads Stock from Quandl.
        Computes daily Returns based on Adj Close.
        Returns pandas dataframe.
        """
        import quandl
        df =  quandl.get(symbol, trim_start = start, trim_end = end, authtoken="your token")

        df.columns.values[-1] = 'AdjClose'
        df.columns = df.columns + '_' + name
        df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
    
        return df


    def loadAllFromWeb(self, fout, start, end):
        """
        Collects predictors data from Yahoo Finance and Quandl.
        Returns a list of dataframes.
        """
        #start = parser.parse(start_string)
        #end = parser.parse(end_string)
    
        #nasdaq = self.getStockFromYahoo('^IXIC', start, end)
        #print('got nasdaq')
        #frankfurt = self.getStockFromYahoo('^GDAXI', start, end)
        #print('got frankfurt')
        #london = self.getStockFromYahoo('^FTSE', start, end)
        #print('got london')
        #paris = self.getStockFromYahoo('^FCHI', start, end)
        #print('got paris')
        #hkong = self.getStockFromYahoo('^HSI', start, end)
        #print('got hkong')
        #nikkei = self.getStockFromYahoo('^N225', start, end)
        #print('got nikkei')
        australia = self.getStockFromYahoo('^AXJO', start, end)
        # need to shift australian values back a date due to time zone dif with US
        for column in australia.columns:
            australia[column] = australia[column].shift(-1)

        print('got australia')
    
        #djia = self.getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start, end) 
        #print('got djia')
    
        out =  data.get_data_yahoo(fout, start, end)
        out.columns.values[-1] = 'AdjClose'
        out.columns = out.columns + '_Out'
        out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
        #trainData['output'] = out
        #trainData['nasdaq'] = nasdaq
        #trainData['djia'] = djia
        #trainData['frankfurt'] = frankfurt
        #trainData['paris'] = paris
        #trainData['hkong'] = hkong
        #trainData['nikkei'] = nikkei
        #trainData['australia'] = australia

        return out, australia


    def returnDataForClassification(self, start_test):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """

        dataset = self.finalTrainingData

        le = preprocessing.LabelEncoder()
    
        dataset['UpDown'] = dataset['Return_Out']
        dataset.UpDown[dataset.Return_Out >= 0] = 'Up'
        dataset.UpDown[dataset.Return_Out < 0] = 'Down'
        dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
        features = dataset.columns[1:-1]
        X = dataset[features]    
        y = dataset.UpDown
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)    
    
        #X_train = X[X.index < start_test]
        #y_train = y[y.index < start_test]              
    
        #X_test = X[X.index >= start_test]    
        #y_test = y[y.index >= start_test]
    
        return X_train, y_train, X_test, y_test   

