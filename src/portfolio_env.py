import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, returns_df, initial_cash=100000):
        super().__init__()
        self.returns = returns_df.values
        self.n_assets = self.returns.shape[1]
        self.initial_cash = initial_cash
        self.current_step = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets * 2 + 1,))
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.volatility = np.std(self.returns[:10], axis=0).mean() if len(self.returns) > 10 else 0.01
        return self._get_obs(), {}

    def _get_obs(self):
        current_returns = self.returns[self.current_step]
        return np.concatenate([current_returns, self.weights, [self.volatility]])

    def step(self, action):
        action = np.clip(action, 0, 1)
        self.weights = action / np.sum(action) if np.sum(action) > 0 else self.weights
        port_return = np.dot(self.returns[self.current_step], self.weights)
        self.portfolio_value *= (1 + port_return)
        self.volatility = np.std(self.returns[max(0, self.current_step-10):self.current_step+1], axis=0).mean()
        reward = port_return - 0.5 * self.volatility  # Sharpe-like reward
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self._get_obs(), reward, done, False, {'portfolio_value': self.portfolio_value}