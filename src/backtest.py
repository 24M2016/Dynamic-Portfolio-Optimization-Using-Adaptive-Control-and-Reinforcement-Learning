def backtest_model(returns, model_path='ppo_portfolio'):
    from stable_baselines3 import PPO
    env = PortfolioEnv(returns)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    portfolio_values = [env.portfolio_value]
    weights_history = [env.weights.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        weights_history.append(env.weights.copy())
    return portfolio_values, weights_history