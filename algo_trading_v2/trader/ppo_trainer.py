"""
trader/ppo_trainer.py
----------------------
Trenuje agenta PPO (stable-baselines3) z parametrami dostarczonymi przez Optuna.

Przepływ:
    Optuna → reward_config → PPOTrainer → PPO uczy się → zwraca metryki → Optuna
"""

import logging
import os
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Trenuje PPO na danych historycznych z podanym reward_config.
    Wywoływany przez MetaAgent dla każdego trialu Optuna.
    """

    def __init__(
        self,
        candles:          list,
        feature_builder,
        reward_engine,
        extra_signals:    dict,
        initial_balance:  float = 1000.0,
        commission_pct:   float = 0.000,
        n_episodes:       int   = 10,
        progress_callback: Optional[Callable] = None,
        locked_features:  dict = None,
    ):
        self.candles          = candles
        self.fb               = feature_builder
        self.re               = reward_engine
        self.extra_signals    = extra_signals
        self.initial_balance  = initial_balance
        self.commission_pct   = commission_pct
        self.n_episodes       = n_episodes
        self.progress_cb      = progress_callback
        self.locked_features  = locked_features or {}

        self._model  = None
        self._stop   = False

    def stop(self):
        self._stop = True

    def train_and_evaluate(self) -> dict:
        """
        Trenuje PPO i zwraca metryki.
        Wywoływany przez MetaAgent._run_trial().
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from trader.ppo_env import PPOTradingEnv

        # Oblicz total_timesteps
        usable_candles   = len(self.candles) - 50  # warmup
        steps_per_ep     = usable_candles
        total_timesteps  = steps_per_ep * self.n_episodes

        if total_timesteps < 100:
            logger.warning("PPOTrainer: za mało świec")
            return self._empty_metrics()

        # Stwórz środowisko
        env = PPOTradingEnv(
            candles         = self.candles,
            feature_builder = self.fb,
            reward_engine   = self.re,
            extra_signals   = self.extra_signals,
            initial_balance = self.initial_balance,
            commission_pct  = self.commission_pct,
        )

        trainer_ref = self

        class StopCallback(BaseCallback):
            def __init__(self_):
                super().__init__()
                self_.ep_count   = 0
                self_.last_stats = {}

            def _on_step(self_) -> bool:
                return not trainer_ref._stop

            def _on_rollout_end(self_):
                self_.ep_count += 1
                # Pobierz aktualne statystyki ze środowiska
                try:
                    stats = env.get_episode_stats()
                    self_.last_stats = stats
                except Exception:
                    pass
                if trainer_ref.progress_cb:
                    trainer_ref.progress_cb(
                        self_.ep_count,
                        trainer_ref.n_episodes,
                        self_.last_stats,
                    )

        # PPO model
        # n_steps musi być <= liczby kroków w epizodzie
        n_steps    = min(512, steps_per_ep)
        # batch_size musi być dzielnikiem n_steps
        batch_size = 64
        while n_steps % batch_size != 0 and batch_size > 8:
            batch_size = batch_size // 2

        self._model = PPO(
            policy        = "MlpPolicy",
            env           = env,
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = n_steps,
            batch_size    = batch_size,
            n_epochs      = 5,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            policy_kwargs = dict(
                net_arch = [256, 256],
            ),
        )

        try:
            callback = StopCallback()
            self._model.learn(
                total_timesteps = total_timesteps,
                callback        = callback,
                reset_num_timesteps = True,
            )
        except Exception as e:
            logger.error(f"PPOTrainer: błąd treningu — {e}")
            return self._empty_metrics()

        # Ewaluacja po treningu — uruchom jeden pełny epizod
        return self._evaluate(env)

    def _evaluate(self, env) -> dict:
        """Uruchamia wytrenowany model na danych i zwraca metryki."""
        if self._model is None:
            return self._empty_metrics()

        obs, _ = env.reset()
        done   = False

        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                break

        return env.get_episode_stats()

    def save_model(self, path: str):
        """Zapisuje wytrenowany model PPO."""
        if self._model:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            self._model.save(path)
            logger.info(f"PPOTrainer: model zapisany → {path}")

    def load_model(self, path: str) -> bool:
        """Ładuje zapisany model PPO."""
        try:
            from stable_baselines3 import PPO
            from trader.ppo_env import PPOTradingEnv

            env = PPOTradingEnv(
                candles         = self.candles,
                feature_builder = self.fb,
                reward_engine   = self.re,
                extra_signals   = self.extra_signals,
                initial_balance = self.initial_balance,
                commission_pct  = self.commission_pct,
            )
            self._model = PPO.load(path, env=env)
            logger.info(f"PPOTrainer: model załadowany ← {path}")
            return True
        except Exception as e:
            logger.error(f"PPOTrainer: błąd ładowania — {e}")
            return False

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_return_pct": 0.0,
            "final_balance":    0.0,
            "total_trades":     0,
            "win_rate":         0.0,
            "sharpe":           0.0,
            "max_drawdown_pct": 0.0,
        }