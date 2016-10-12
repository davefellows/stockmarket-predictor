import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from datetime import datetime
#from constants import constants
import pytest
import pandas as pd
from trainingData import trainingData

predictionSymbol = '^GSPC' #symbol for S&P500
startdate = datetime(2016,1,1)
enddate = datetime(2016,1,31)
index = 4
delta = range(2, 3) 
lags = range(2, 3)

#def test_getStockFromYahoo():
#    td = trainingData()
#    dataframe = td.getStockFromYahoo('MSFT', startdate, enddate)
#    assert dataframe.index.size == 19

#def test_getStockFromQuandl():
#    td = trainingData()
#    dataframe = td.getStockFromYahoo('MSFT', startdate, enddate)
#    assert dataframe.index.size == 19

#def test_loadAllFromWeb():
#    td = trainingData()
#    trainData = td.loadAllFromWeb(predictionSymbol, startdate, enddate)
#    assert len(trainData) == 7
#    assert trainData[0].index.size == 19

def test_loadAndApplyTrainingData():
    td = trainingData()
    td.load(predictionSymbol, startdate, enddate, delta, lags)
    print(td.mergedTrainingData.values)