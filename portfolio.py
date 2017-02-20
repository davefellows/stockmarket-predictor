import pandas as pd
import backtest
from backtest import Portfolio

class MarketIntradayPortfolio(Portfolio):
    """Buys or sells 500 shares of an asset at the opening price of
    every bar, depending upon the direction of the forecast, closing 
    out the trade at the close of the bar.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0, shares=500):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        """Generate the positions DataFrame, based on the signals
        provided by the 'signals' DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)

        positions[self.symbol] = self.shares*self.signals['signal']
        return positions
                    
    def backtest_portfolio(self):
        """Backtest the portfolio and return a DataFrame containing
        the equity curve and the percentage returns."""
       
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()
            
        portfolio['price_diff'] = self.bars['Close']-self.bars['Open']
        portfolio['price_diff'][0:5] = 0.0
        #take 10% fudge factor off for loss of efficiency on open/close pricing and brokerage fees
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff'] #* 0.9
     
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()

        return portfolio