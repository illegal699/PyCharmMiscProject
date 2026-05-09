"""
trader/ppo_env.py
------------------
Środowisko gymnasium-compatible dla PPO (stable-baselines3).
Opakowuje TraderEnv w standardowy interfejs gymnasium.Env.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class PPOTradingEnv(gym.Env):
    """
    Środowisko tradingowe zgodne z gymnasium API.
    Używane przez PPO z stable-baselines3.
    
    Observation: znormalizowany wektor cech (float32)
    Action:      Discrete(4) — HOLD=0, LONG=1, SHORT=2, CLOSE=3
    Reward:      z RewardEngine (zależny od reward_config)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        candles:         list,
        feature_builder,
        reward_engine,
        extra_signals:   dict = None,
        initial_balance: float = 1000.0,
        commission_pct:  float = 0.001,
    ):
        super().__init__()

        self.candles         = candles
        self.fb              = feature_builder
        self.re              = reward_engine
        self.extra_signals   = extra_signals or {}
        self.initial_balance = initial_balance
        self.commission_pct  = commission_pct

        # Liczba cech
        n_features = feature_builder.cfg.feature_count()

        # Przestrzenie
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  = +1.0,
            shape = (n_features,),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Wewnętrzne środowisko
        self._env = None
        self._last_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        from trader.trader_env import TraderEnv
        self._env = TraderEnv(
            candles         = self.candles,
            feature_builder = self.fb,
            reward_engine   = self.re,
            extra_signals   = self.extra_signals,
            initial_balance = self.initial_balance,
            commission_pct  = self.commission_pct,
        )
        obs = self._env.reset()
        self._last_obs = np.array(obs, dtype=np.float32)
        return self._last_obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        self._last_obs = np.array(obs, dtype=np.float32)
        truncated = False
        return self._last_obs, float(reward), done, truncated, info

    def get_episode_stats(self) -> dict:
        """Zwraca statystyki epizodu po zakończeniu."""
        if self._env:
            return self._env.get_episode_stats()
        return {}

    def render(self):
        pass

    def close(self):
        pass
