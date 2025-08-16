from stable_baselines3.common.callbacks import BaseCallback

class AdaptiveCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.vol_threshold = 0.02

    def _on_step(self):
        vol = self.locals['infos'][0].get('volatility', 0.01)
        self.model.learning_rate = 1e-4 if vol > self.vol_threshold else 3e-4
        return True