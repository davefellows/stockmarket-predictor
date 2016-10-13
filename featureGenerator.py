import pandas as pd

index = 6 # columns to remove from each dataset

def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n) #pd.Series(dataframe[returns]).rolling(window=n).mean()


def applyRollMeanDelayedReturns(datasets, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    
    return datasets    


def mergeDataframes(datasets, cut):
    """
    merges datasets in the list 
    """
    #subset = []tion
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    
    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
    # don't need to cut as only using relatively recent data for training
    #finance = finance[finance.index > cut]
    return finance


def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lags list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    
    # need to shift the y value back a day so we're predicting
    # next day price, rather than today!
    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:].dropna()