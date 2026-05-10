"""
trader/trainer.py
------------------
Pętla treningu agenta RL (PPO) dla Algorytmu #3.
Używa stable-baselines3 jeśli dostępne, fallback na prosty Q-learning.

Instalacja: pip install stable-baselines3 gymnasium
"""

import logging
import os
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Konfiguracja sesji treningu."""
    symbol:        str   = "BTCUSDT"
    interval:      str   = "5m"
    date_from:     str   = "2024-01-01"
    date_to:       str   = "2024-12-31"
    n_episodes:    int   = 100
    initial_balance: float = 1000.0
    commission_pct:  float = 0.0004
    model_name:    str   = "trader_ppo"
    save_path:     str   = "models/"


@dataclass
class TrainingProgress:
    """Stan postępu treningu — przekazywany do UI."""
    episode:           int   = 0
    total_episodes:    int   = 0
    pct_complete:      float = 0.0
    current_return:    float = 0.0
    best_return:       float = -999.0
    avg_return_10:     float = 0.0
    win_rate:          float = 0.0
    total_trades:      int   = 0
    sharpe:            float = 0.0
    max_drawdown:      float = 0.0
    status:            str   = "idle"     # idle | running | done | error
    message:           str   = ""
    history:           list  = field(default_factory=list)   # lista wyników per epizod
    start_time:        Optional[float] = None
    elapsed_sec:       float = 0.0
    eta_sec:           float = 0.0


class Trainer:
    """
    Trenuje agenta RL na danych historycznych.

    Używa PPO z stable-baselines3 jeśli dostępne,
    fallback na prosty algorytm gradientowy.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        reward_config,
        feature_config,
        progress_callback: Optional[Callable] = None,
    ):
        self.tc       = training_config
        self.rc       = reward_config
        self.fc       = feature_config
        self.callback = progress_callback
        self.progress = TrainingProgress(total_episodes=training_config.n_episodes)
        self._stop    = False

    def stop(self):
        """Zatrzymuje trening."""
        self._stop = True
        logger.info("Trainer: zatrzymywanie...")

    def run(self) -> TrainingProgress:
        """
        Uruchamia trening.
        Wywołuj w osobnym wątku żeby nie blokować UI.
        """
        self.progress.status     = "running"
        self.progress.start_time = time.time()
        self._stop = False

        try:
            # Pobierz dane historyczne
            self._update("📥 Pobieranie danych historycznych z Binance...")
            candles = self._fetch_candles()
            if not candles:
                raise RuntimeError("Brak danych historycznych")

            self._update(f"✅ Pobrano {len(candles)} świec dla {self.tc.symbol} {self.tc.interval}")

            # Zbuduj środowisko
            from trader.reward_engine   import RewardEngine
            from trader.feature_builder import FeatureBuilder
            from trader.trader_env      import TraderEnv

            reward_engine   = RewardEngine(self.rc)
            feature_builder = FeatureBuilder(self.fc)

            self._update(f"🔧 Środowisko: {self.fc.feature_count()} cech wejściowych")

            # Próbuj PPO (stable-baselines3)
            use_ppo = self._check_sb3()

            if use_ppo:
                self._update("🤖 Model: PPO (stable-baselines3)")
                result = self._train_ppo(candles, reward_engine, feature_builder)
            else:
                self._update("🤖 Model: Simple Policy Gradient (fallback)")
                result = self._train_simple(candles, reward_engine, feature_builder)

            self.progress.status  = "done"
            self.progress.message = f"✅ Trening zakończony! Najlepszy zwrot: {self.progress.best_return:.1f}%"
            self._notify()

        except Exception as e:
            self.progress.status  = "error"
            self.progress.message = f"❌ Błąd: {str(e)}"
            logger.error(f"Trainer: błąd — {e}", exc_info=True)
            self._notify()

        return self.progress

    def _train_simple(self, candles, reward_engine, feature_builder) -> None:
        """
        Prosty algorytm Policy Gradient bez zewnętrznych bibliotek.
        Używa epizodów na danych historycznych.
        """
        from trader.trader_env import TraderEnv, ACTION_HOLD, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE
        import random

        n_actions    = 4
        n_features   = feature_builder.cfg.feature_count()
        recent_returns = []

        # Prosta sieć: wagi dla każdej cechy × akcji
        weights = [[0.0] * n_features for _ in range(n_actions)]
        lr      = 0.01

        for ep in range(self.tc.n_episodes):
            if self._stop:
                self.progress.message = "⏹️ Trening zatrzymany przez użytkownika"
                break

            # Każdy epizod przechodzi przez WSZYSTKIE świece chronologicznie
            # jak w rzeczywistości — od pierwszej do ostatniej
            env   = TraderEnv(candles, feature_builder, reward_engine,
                              initial_balance=self.tc.initial_balance,
                              commission_pct=self.tc.commission_pct)
            obs   = env.reset()
            done  = False
            ep_reward = 0.0
            ep_log    = []

            while not done:
                # Prosta polityka: wybierz akcję z najwyższym score
                scores = [sum(w * o for w, o in zip(weights[a], obs)) for a in range(n_actions)]

                # Epsilon-greedy exploration (maleje z czasem)
                epsilon = max(0.05, 0.5 - ep / self.tc.n_episodes)
                if random.random() < epsilon:
                    action = random.randint(0, n_actions - 1)
                else:
                    action = scores.index(max(scores))

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_log.append((action, reward))

                # Policy gradient update
                for a in range(n_actions):
                    sign = 1.0 if a == action else -0.1
                    for i in range(n_features):
                        if i < len(obs):
                            weights[a][i] += lr * reward * sign * obs[i]

            # Zbierz statystyki
            stats = env.get_episode_stats()
            recent_returns.append(stats["total_return_pct"])
            if len(recent_returns) > 10:
                recent_returns.pop(0)

            self._update_progress(ep, stats, recent_returns)

        self._save_weights(weights)

    def _train_ppo(self, candles, reward_engine, feature_builder) -> None:
        """Trening PPO przez stable-baselines3."""
        try:
            import gymnasium as gym
            import numpy as np
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from trader.trader_env import TraderEnv

            class SB3Env(gym.Env):
                def __init__(self_, candles, fb, re, tc):
                    super().__init__()
                    self_.env = TraderEnv(candles, fb, re,
                                          initial_balance=tc.initial_balance,
                                          commission_pct=tc.commission_pct)
                    n = fb.cfg.feature_count()
                    self_.observation_space = gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(n,), dtype=np.float32
                    )
                    self_.action_space = gym.spaces.Discrete(4)

                def reset(self_, **kwargs):
                    obs = self_.env.reset()
                    return np.array(obs, dtype=np.float32), {}

                def step(self_, action):
                    obs, reward, done, info = self_.env.step(action)
                    return np.array(obs, dtype=np.float32), reward, done, False, info

            trainer_ref = self

            class ProgressCallback(BaseCallback):
                def __init__(self_):
                    super().__init__()
                    self_.ep_count = 0

                def _on_step(self_) -> bool:
                    if trainer_ref._stop:
                        return False
                    return True

                def _on_rollout_end(self_):
                    self_.ep_count += 1
                    ep = min(self_.ep_count, trainer_ref.tc.n_episodes)
                    pct = ep / trainer_ref.tc.n_episodes * 100
                    trainer_ref.progress.episode      = ep
                    trainer_ref.progress.pct_complete = pct
                    trainer_ref._update(f"Epizod {ep}/{trainer_ref.tc.n_episodes}  ({pct:.0f}%)")

            sb3_env = SB3Env(candles, feature_builder, reward_engine, self.tc)
            model   = PPO("MlpPolicy", sb3_env, verbose=0,
                          n_steps=512, batch_size=64, n_epochs=10)

            total_steps = len(candles) * self.tc.n_episodes
            model.learn(total_timesteps=total_steps, callback=ProgressCallback())

            os.makedirs(self.tc.save_path, exist_ok=True)
            path = os.path.join(self.tc.save_path, self.tc.model_name)
            model.save(path)
            self._update(f"💾 Model zapisany: {path}")

        except ImportError as e:
            logger.warning(f"PPO niedostępny: {e} — używam fallback")
            self._train_simple(candles, reward_engine, feature_builder)

    def _fetch_candles(self):
        """Pobiera dane historyczne z Binance w wielu requestach."""
        import sys, os, time, requests
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from trend_analyzer.data_fetcher import BinanceDataFetcher, Candle

        fetcher  = BinanceDataFetcher()
        interval = self.tc.interval
        symbol   = self.tc.symbol

        from datetime import datetime, timezone
        try:
            dt_from = datetime.strptime(self.tc.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dt_to   = datetime.strptime(self.tc.date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            self._update("Nieprawidłowy format dat — pobieranie ostatnich 1000 świec")
            return fetcher.get_candles(symbol, interval, limit=1000)

        iv_ms    = self._interval_minutes() * 60 * 1000
        start_ms = int(dt_from.timestamp() * 1000)
        end_ms   = int(dt_to.timestamp()   * 1000)
        all_candles = []
        batch_size  = 1000
        current_ms  = start_ms

        self._update(f"Pobieranie danych {symbol} {interval} od {self.tc.date_from} do {self.tc.date_to}...")

        while current_ms < end_ms:
            if self._stop:
                break
            try:
                import os as _os
                api_key = _os.getenv("BINANCE_API_KEY", "")
                headers = {"X-MBX-APIKEY": api_key} if api_key else {}
                r = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={
                        "symbol":    symbol,
                        "interval":  interval,
                        "startTime": current_ms,
                        "endTime":   end_ms,
                        "limit":     batch_size,
                    },
                    headers=headers,
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
                if not data:
                    break
                batch = [Candle(row) for row in data]
                all_candles.extend(batch)
                current_ms = batch[-1].timestamp + iv_ms
                self._update(f"Pobrano {len(all_candles)} świec...")
                if len(data) < batch_size:
                    break
                time.sleep(0.3)  # szanuj rate limit
            except Exception as e:
                self._update(f"Błąd pobierania: {e} — używam pobranych świec ({len(all_candles)})")
                break

        if not all_candles:
            self._update("Fallback: pobieranie ostatnich 1000 świec")
            return fetcher.get_candles(symbol, interval, limit=1000)

        self._update(f"✅ Łącznie pobrano {len(all_candles)} świec")
        return all_candles

    def _interval_minutes(self) -> int:
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        return mapping.get(self.tc.interval, 5)

    def _check_sb3(self) -> bool:
        try:
            import stable_baselines3
            import gymnasium
            return True
        except ImportError:
            return False

    def _update(self, message: str):
        self.progress.message = message
        if self.progress.start_time:
            self.progress.elapsed_sec = time.time() - self.progress.start_time
        logger.info(f"Trainer: {message}")
        self._notify()

    def _update_progress(self, ep: int, stats: dict, recent_returns: list):
        p = self.progress
        p.episode         = ep + 1
        p.pct_complete    = (ep + 1) / p.total_episodes * 100
        p.current_return  = stats["total_return_pct"]
        p.best_return     = max(p.best_return, stats["total_return_pct"])
        p.avg_return_10   = sum(recent_returns) / len(recent_returns)
        p.win_rate        = stats["win_rate"]
        p.total_trades    = stats["total_trades"]
        p.sharpe          = stats["sharpe"]
        p.max_drawdown    = stats["max_drawdown_pct"]
        p.elapsed_sec     = time.time() - p.start_time if p.start_time else 0

        if p.episode > 1:
            sec_per_ep = p.elapsed_sec / p.episode
            p.eta_sec  = sec_per_ep * (p.total_episodes - p.episode)

        p.history.append({
            "episode":    ep + 1,
            "return_pct": stats["total_return_pct"],
            "win_rate":   stats["win_rate"],
            "trades":     stats["total_trades"],
            "sharpe":     stats["sharpe"],
        })

        p.message = (
            f"Epizod {p.episode}/{p.total_episodes} | "
            f"Return: {p.current_return:+.1f}% | "
            f"Best: {p.best_return:+.1f}% | "
            f"WR: {p.win_rate:.0f}% | "
            f"Sharpe: {p.sharpe:.2f}"
        )
        self._notify()

    def _save_weights(self, weights):
        os.makedirs(self.tc.save_path, exist_ok=True)
        path = os.path.join(self.tc.save_path, f"{self.tc.model_name}_weights.json")
        with open(path, "w") as f:
            json.dump(weights, f)
        self._update(f"💾 Wagi zapisane: {path}")

    def _notify(self):
        if self.callback:
            try:
                self.callback(self.progress)
            except Exception:
                pass
