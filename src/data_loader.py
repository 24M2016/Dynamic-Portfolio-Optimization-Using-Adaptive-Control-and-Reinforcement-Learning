import yfinance as yf
import pandas as pd

def load_data(tickers=['AAPL', 'MSFT', 'GOOG'], start='2010-01-01', end='2025-08-16'):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data.to_csv('datasets/portfolio_data.csv')
    returns = data.pct_change().dropna()
    return returns