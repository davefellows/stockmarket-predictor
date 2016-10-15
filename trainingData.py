import featureGenerator as fg
import pandas as pd
from pandas_datareader import data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

pd.options.mode.chained_assignment = None  # default='warn'

class trainingData(object):

    def load(self, fout, startdate, enddate, delta, lags):
        """
        Loads all data from web into dataFrames, 
        adds various features, merges datasets.
        """
        self.trainData = self.loadAllFromWeb(fout, startdate, enddate)
        self.trainDataWithFeatures = fg.applyRollMeanDelayedReturns(self.trainData, delta)
        self.mergedData = fg.mergeDataframes(self.trainDataWithFeatures, enddate)
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
    
        nasdaq = self.getStockFromYahoo('^IXIC', start, end)
        print('got nasdaq')
        sp500 = self.getStockFromYahoo('^GSPC', start, end)
        print('got sp500')
        volatility = self.getStockFromYahoo('^VIX', start, end)
        print('got volatility')
        australia = self.getStockFromYahoo('^AXJO', start, end)
        print('got australia')
        hkong = self.getStockFromYahoo('^HSI', start, end)
        print('got hkong')

        #london = self.getStockFromYahoo('^FTSE', start, end)
        #print('got london')
        #gold = self.getStockFromYahoo('^XAU', start, end)
        #print('got gold')
        #eur = self.getStockFromYahoo('EURUSD=X', start, end)
        #print('got euro')

        #frankfurt = self.getStockFromYahoo('^GDAXI', start, end)
        #print('got frankfurt')
        #paris = self.getStockFromYahoo('^FCHI', start, end)
        #print('got paris')
        #nikkei = self.getStockFromYahoo('^N225', start, end)
        #print('got nikkei')

        # Maybe we don't need this as dates are regionalized?
        #if fout == '^AXJO':
        #    # if we're predicting Australian index then need 
        #    # to shift other values forward to align with time zone
        #    for column in nasdaq.columns:
        #        nasdaq[column] = nasdaq[column].shift(1)
        #    for column in sp500.columns:
        #        sp500[column] = sp500[column].shift(1)
        #    for column in volatility.columns:
        #        volatility[column] = volatility[column].shift(1)
        #else:
        #    # else shift AUS back to align with others
        #    for column in australia.columns:
        #        australia[column] = australia[column].shift(-1)
        #    for column in hkong.columns:
        #        hkong[column] = hkong[column].shift(-1)
            
    
        #djia = self.getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start, end) 
        #print('got djia')
    
        out =  data.get_data_yahoo(fout, start, end)
        out.columns.values[-1] = 'AdjClose'
        out.columns = out.columns + '_Out'
        out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
        return out, australia, nasdaq, sp500, volatility, hkong

    def normalizeData(self, train_data, test_data=None):
        scaler = StandardScaler()  
        # fit only on training data
        scaler.fit(train_data)  
        train_data = scaler.transform(train_data)  
        if test_data is not None:
            test_data = scaler.transform(test_data)
            return train_data, test_data
        
        return train_data


    def returnDataForClassification(self, start_test, test_size=0.2):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """

        dataset = self.finalTrainingData.dropna()

        le = preprocessing.LabelEncoder()
    
        dataset['UpDown'] = dataset['Return_Out']
        dataset.UpDown[dataset.Return_Out >= 0] = 'Up'
        dataset.UpDown[dataset.Return_Out < 0] = 'Down'
        dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
        features = dataset.columns[1:-1]
        X = dataset[features]    
        y = dataset.UpDown
        
        #X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)    

        # apply same transformation to training and test data
        #X_train, X_test = self.normalizeData(X_train, X_test)
    
        return X_train, y_train, X_test, y_test   

    def returnDataForBacktesting(self):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """

        dataset = self.finalTrainingData.dropna()

        le = preprocessing.LabelEncoder()
    
        dataset['UpDown'] = dataset['Return_Out']
        dataset.UpDown[dataset.Return_Out >= 0] = 'Up'
        dataset.UpDown[dataset.Return_Out < 0] = 'Down'
        dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
        features = dataset.columns[1:-1]
        X = dataset[features] 
        #X = StandardScaler().fit_transform(X)
        
        return X #self.normalizeData(X)

    def returnDataForPrediction(self):
        """
        generates categorical output column, attach to dataframe 
        label the categories and split into train and test
        """

        dataset = self.mergedData

        dataset['UpDown'] = dataset['Return_Out']
    
        features = dataset.columns[1:-1]
        #X = dataset[features]    
        
        return X #self.normalizeData(X)
