# rl/environment.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, config: dict = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.config = config or {}

        # === WSZYSTKIE PARAMETRY Z ZAKŁADKI ===
        self.initial_capital = self.config.get("initial_capital", 10000.0)
        self.position_size_pct = self.config.get("position_size_pct", 10.0)
        self.max_position_pct = self.config.get("max_position_pct", 25.0)
        self.leverage = self.config.get("leverage", 1)
        self.use_futures = self.config.get("use_futures", False)

        self.stop_loss_pct = self.config.get("stop_loss_pct", 3.0)
        self.take_profit_pct = self.config.get("take_profit_pct", 15.0)
        self.max_drawdown_limit = self.config.get("max_drawdown_limit", 15.0)

        # Nagrody i kary (używane w reward)
        self.reward_profit = self.config.get("reward_profit", 1.2)
        self.penalty_drawdown = self.config.get("penalty_drawdown", 2.5)
        self.penalty_transaction = self.config.get("penalty_transaction", 0.15)

        # Stan
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.max_equity = self.initial_capital
        self.current_step = 0

        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0

        self.feature_cols = [col for col in df.columns if col not in
                           ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        self.action_space = spaces.Discrete(3)  # 0=Sell, 1=Hold, 2=Buy
        self.observation_space = spaces.Box(
            low=-10, high=10,
            shape=(len(self.feature_cols) + 5,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.max_equity = self.initial_capital
        self.current_step = 0
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        return self._get_obs(), {}

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, {"equity": self.equity}

        current_price = float(self.df.loc[self.current_step, 'close'])
        done = self.current_step >= len(self.df) - 2
        reward = 0.0

        # === OTWIERANIE POZYCJI ===
        if action == 2 and self.position == 0:      # BUY
            position_value = self.balance * (self.position_size_pct / 100.0)
            position_value = min(position_value, self.balance * (self.max_position_pct / 100.0))
            self.position = (position_value * self.leverage) / current_price
            self.entry_price = current_price
            self.trades.append({"type": "buy", "price": current_price, "step": self.current_step})

        # === ZAMYKANIE POZYCJI (SELL) ===
        elif action == 0 and self.position > 0:
            profit = (current_price - self.entry_price) * self.position
            self.balance += profit
            self.total_profit += profit

            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            self.trades.append({
                "type": "sell",
                "price": current_price,
                "profit": profit,
                "step": self.current_step
            })
            reward += profit * 0.1
            self.position = 0.0

        # === AUTOMATYCZNY STOP-LOSS / TAKE-PROFIT ===
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
            position_value = self.position * current_price

            # Stop Loss
            if unrealized_pnl < 0:
                sl_value = position_value * (self.stop_loss_pct / 100.0)
                if abs(unrealized_pnl) >= sl_value:
                    self.balance += unrealized_pnl
                    self.total_profit += unrealized_pnl
                    self.losing_trades += 1
                    self.position = 0.0
                    reward -= 5.0  # kara za SL

            # Take Profit
            if unrealized_pnl > 0:
                tp_value = position_value * (self.take_profit_pct / 100.0)
                if unrealized_pnl >= tp_value:
                    self.balance += unrealized_pnl
                    self.total_profit += unrealized_pnl
                    self.winning_trades += 1
                    self.position = 0.0
                    reward += 3.0  # bonus za TP

        # === MAX DRAWDOWN ===
        current_drawdown = (self.max_equity - self.equity) / self.max_equity * 100
        if current_drawdown > self.max_drawdown_limit:
            if self.position > 0:
                unrealized = (current_price - self.entry_price) * self.position
                self.balance += unrealized
                self.total_profit += unrealized
                self.position = 0.0
            reward -= 10.0  # duża kara

        self.equity = self.balance + self.position * current_price
        self.max_equity = max(self.max_equity, self.equity)

        reward += self._calculate_reward(current_price)
        self.current_step += 1

        return self._get_obs(), reward, done, False, {"equity": self.equity}

    def _calculate_reward(self, current_price):
        reward = 0.0
        equity_return = (self.equity - self.initial_capital) / self.initial_capital
        reward += equity_return * self.reward_profit
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        reward -= drawdown * self.penalty_drawdown
        return float(reward)

    def _get_obs(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)
        norm_price = row['close'] / self.df['close'].iloc[0]
        position_ratio = (self.position * row['close']) / self.equity if self.equity > 0 else 0
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        obs = np.concatenate([features, [norm_price, position_ratio, drawdown, self.position, self.equity/self.initial_capital]])
        return obs.astype(np.float32)

    def get_stats(self):
        total_trades = len([t for t in self.trades if t["type"] == "sell"])
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        return {
            "final_equity": self.equity,
            "total_return": (self.equity - self.initial_capital) / self.initial_capital * 100,
            "total_profit": self.total_profit,
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "max_drawdown": (self.max_equity - self.equity) / self.max_equity * 100 if self.max_equity > 0 else 0,
        }