import matplotlib.pyplot as plt

def visualize_results(returns, portfolio_values, weights_history, tickers):
    dates = returns.index
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(portfolio_values)-1], portfolio_values[:-1], label='Portfolio Value')
    plt.title('Portfolio Performance (Backtest)')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.savefig('results/plots/portfolio_performance.png')
    plt.close()

    weights_df = pd.DataFrame(weights_history[:-1], columns=tickers, index=dates[:len(weights_history)-1])
    weights_df.plot.area(figsize=(12, 6))
    plt.title('Dynamic Asset Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.savefig('results/plots/asset_weights.png')
    plt.close()