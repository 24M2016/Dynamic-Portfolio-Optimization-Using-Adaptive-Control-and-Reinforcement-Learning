from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from portfolio_env import PortfolioEnv
from adaptive_control import AdaptiveCallback

def train_model(returns, model_path='ppo_portfolio'):
    env = make_vec_env(lambda: PortfolioEnv(returns), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    callback = AdaptiveCallback()
    model.learn(total_timesteps=100000, callback=callback)
    model.save(model_path)