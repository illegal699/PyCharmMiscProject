# rl/agent.py
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self, env, algorithm: str = "PPO"):
        self.env = env
        self.algorithm = algorithm.upper()
        self.model = None

    def create_model(self):
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gae_lambda=0.95,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        logger.info(f"✅ Utworzono model {self.algorithm}")

    def train(self, total_timesteps: int = 100_000, save_path: str = "models/", save_freq: int = 20000):
        os.makedirs(save_path, exist_ok=True)

        class SaveCallback(BaseCallback):
            def __init__(self, save_freq, save_path):
                super().__init__()
                self.save_freq = save_freq
                self.save_path = save_path

            def _on_step(self):
                if self.n_calls % self.save_freq == 0:
                    path = os.path.join(self.save_path, f"model_{self.n_calls}.zip")
                    self.model.save(path)
                    logger.info(f"Model zapisany: {path}")
                return True

        callback = SaveCallback(save_freq=save_freq, save_path=save_path)

        # Bez progress_bar=True
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        final_path = os.path.join(save_path, "final_model.zip")
        self.model.save(final_path)
        logger.info(f"✅ Trening zakończony! Model zapisany: {final_path}")

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model wczytany z: {path}")