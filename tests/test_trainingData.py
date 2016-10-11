import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import pandas as pd
from trainingData import trainingData

def test_one():
    x = 'this'
    assert 'h' in x

def test_getStockFromYahoo():
    td = trainingData()
    dataframe = td.getStockFromYahoo('MSFT', '2016-01-01', '2016-01-02')
    print (dataframe)



class test_trainingData:
    """Tests for the trainingData class"""

    def test_one(self):
        x = 'this'
        assert 'h' in x

    def test_getStockFromYahoo(self):
        dataframe = trainingData.getStockFromYahoo('MSFT', '2016-01-01', '2016-01-02')
        print (dataframe)
        
