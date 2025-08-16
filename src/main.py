import os
from data_loader import load_data
from train_model import train_model
from backtest import backtest_model
from visualize import visualize_results

if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    tickers = ['AAPL', 'MSFT', 'GOOG']
    returns = load_data(tickers)
    train_returns = returns.loc[:'2020-01-01']
    test_returns = returns.loc['2020-01-02':]

    train_model(train_returns)
    portfolio_values, weights_history = backtest_model(test_returns)
    visualize_results(test_returns, portfolio_values, weights_history, tickers)

    cum_return = (portfolio_values[-1] - 100000) / 100000
    sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252)
    print(f"Cumulative Return: {cum_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")