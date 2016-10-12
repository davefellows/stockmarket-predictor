import featureGenerator as fg
import pandas as pd
#import pandas.io.data
from pandas_datareader import data
from sklearn import preprocessing

class trainingData(object):

    index = 4 # number of datasets?
    lags = range(2, 3)

    def load(self, fout, startdate, enddate, delta):
        """
        Loads all data from web into dataFrames, 
        adds various features, merges datasets.
        """
        trainData = self.loadAllFromWeb(fout, startdate, enddate)
        trainDataWithFeatures = fg.applyRollMeanDelayedReturns(trainData, delta)
        mergedData = fg.mergeDataframes(trainDataWithFeatures, self.index, enddate)
        self.mergedTrainingData = fg.applyTimeLag(mergedData, self.lags, delta)


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
    
        nasdaq = self.getStockFromYahoo('^IXIC', start, end)
        print('got nasdaq')
        frankfurt = self.getStockFromYahoo('^GDAXI', start, end)
        print('got frankfurt')
        #london = self.getStockFromYahoo('^FTSE', start, end)
        #print('got london')
        paris = self.getStockFromYahoo('^FCHI', start, end)
        print('got paris')
        hkong = self.getStockFromYahoo('^HSI', start, end)
        print('got hkong')
        nikkei = self.getStockFromYahoo('^N225', start, end)
        print('got nikkei')
        australia = self.getStockFromYahoo('^AXJO', start, end)
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

        return out, nasdaq, frankfurt, paris, hkong, nikkei, australia


    def returnDataForClassification(self, start_test):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """

        dataset = self.mergedTrainingData

        le = preprocessing.LabelEncoder()
    
        #dataset['UpDown'] = dataset['Return_Out']
        dataset.UpDown[dataset.Return_Out >= 0] = 'Up'
        dataset.UpDown[dataset.Return_Out < 0] = 'Down'
        dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
        features = dataset.columns[1:-1]
        X = dataset[features]    
        y = dataset.UpDown    
    
        X_train = X[X.index < start_test]
        y_train = y[y.index < start_test]              
    
        X_test = X[X.index >= start_test]    
        y_test = y[y.index >= start_test]
    
        return X_train, y_train, X_test, y_test   

