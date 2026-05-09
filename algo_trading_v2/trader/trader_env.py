"""
trader/trader_env.py
---------------------
Środowisko gymnasium dla agenta RL Algorytmu #3.
Symuluje handel na danych historycznych OHLCV + sygnały z Algo #1 i #2.
"""

import logging
import math
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

# Akcje agenta
ACTION_HOLD  = 0
ACTION_LONG  = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3
ACTIONS = ["hold", "long", "short", "close"]


class TraderEnv:
    """
    Środowisko tradingowe dla agenta RL.

    Observation: wektor cech z FeatureBuilder
    Action space: [hold, long, short, close]
    Reward: obliczany przez RewardEngine
    """

    def __init__(
        self,
        candles:        list,           # lista Candle z BinanceDataFetcher
        feature_builder,                # FeatureBuilder
        reward_engine,                  # RewardEngine
        extra_signals:  dict = None,    # sygnały z Algo #1 i #2 per timestamp
        initial_balance: float = 1000.0,
        commission_pct:  float = 0.001, # 0.1% prowizja
    ):
        self.candles         = candles
        self.fb              = feature_builder
        self.re              = reward_engine
        self.extra_signals   = extra_signals or {}
        self.initial_balance = initial_balance
        self.commission_pct  = commission_pct

        # Stan
        self.reset()

    def reset(self) -> list[float]:
        """Resetuje środowisko do stanu początkowego."""
        self.step_idx            = 50        # start po 50 świecach (potrzebne dla wskaźników)
        self.balance             = self.initial_balance
        self.position_side       = None      # "long" | "short" | None
        self.position_entry_price = 0.0
        self.candles_in_position  = 0
        self.candles_since_close  = 999
        self.cumulative_cvd       = 0.0

        # Historia epizodu
        self.trades:     list = []
        self.rewards:    list = []
        self.equity_curve: list = [self.initial_balance]

        return self._observe()

    def step(self, action: int) -> tuple[list, float, bool, dict]:
        """
        Wykonuje krok środowiska.
        Zwraca: (observation, reward, done, info)
        """
        if self.step_idx >= len(self.candles) - 1:
            return self._observe(), 0.0, True, {}

        candle      = self.candles[self.step_idx]
        next_candle = self.candles[self.step_idx + 1]
        action_str  = ACTIONS[action]
        cfg         = self.re.cfg

        # Pobierz sygnały zewnętrzne dla tego kroku
        ext = self._get_external_signals()

        # Oblicz P&L aktualnej pozycji
        pnl_pct = self._calc_pnl(candle.close)

        # Sprawdź stop loss / take profit
        hit_sl = False
        hit_tp = False
        if self.position_side:
            if pnl_pct <= -cfg.stop_loss_pct:
                action_str = "close"
                action     = ACTION_CLOSE
                hit_sl     = True
            elif pnl_pct >= cfg.take_profit_pct:
                action_str = "close"
                action     = ACTION_CLOSE
                hit_tp     = True

        # Sprawdź czy jest sygnał dywergencji
        div_score    = ext.get("divergence_score", 0.0)
        div_strength = ext.get("divergence_strength", "none")
        missed_signal = self._check_missed_signal(action_str, ext)

        # Wykonaj akcję
        trade_info = self._execute_action(action_str, candle, next_candle)

        # Oblicz nagrodę
        reward, breakdown = self.re.compute(
            action              = action_str,
            pnl_pct             = pnl_pct,
            position_side       = self.position_side if action_str != "close" else
                                  trade_info.get("closed_side"),
            candles_in_position = self.candles_in_position,
            candles_since_close = self.candles_since_close,
            trend_score         = ext.get("trend_score_15m", 0.0),
            divergence_score    = div_score,
            divergence_strength = div_strength,
            hit_stop_loss       = hit_sl,
            hit_take_profit     = hit_tp,
            missed_signal       = missed_signal,
        )

        self.rewards.append(reward)
        self.equity_curve.append(self.balance)
        self.step_idx += 1

        if self.position_side:
            self.candles_in_position += 1
        else:
            self.candles_since_close += 1

        done = self.step_idx >= len(self.candles) - 1

        info = {
            "action":     action_str,
            "price":      candle.close,
            "pnl_pct":    pnl_pct,
            "reward":     reward,
            "breakdown":  breakdown,
            "balance":    self.balance,
            "trade_info": trade_info,
        }

        return self._observe(), reward, done, info

    def _execute_action(self, action: str, candle, next_candle) -> dict:
        """Wykonuje akcję i aktualizuje stan pozycji."""
        info = {}
        price = candle.close

        if action == "long" and self.position_side is None:
            cost = self.balance * self.commission_pct   # prowizja od kapitału
            self.balance            -= cost
            self.position_side       = "long"
            self.position_entry_price = price
            self.candles_in_position  = 0
            self.candles_since_close  = 0
            self.trades.append({"side": "long", "entry": price, "step": self.step_idx})
            info = {"opened": "long", "entry_price": price}

        elif action == "short" and self.position_side is None:
            cost = self.balance * self.commission_pct   # prowizja od kapitału
            self.balance            -= cost
            self.position_side       = "short"
            self.position_entry_price = price
            self.candles_in_position  = 0
            self.candles_since_close  = 0
            self.trades.append({"side": "short", "entry": price, "step": self.step_idx})
            info = {"opened": "short", "entry_price": price}

        elif action == "close" and self.position_side is not None:
            pnl_pct = self._calc_pnl(price)
            pnl_abs = self.balance * pnl_pct
            cost    = self.balance * self.commission_pct   # prowizja od kapitału
            self.balance += pnl_abs - cost

            info = {
                "closed_side": self.position_side,
                "entry_price": self.position_entry_price,
                "exit_price":  price,
                "pnl_pct":     pnl_pct,
                "pnl_abs":     pnl_abs,
                "duration":    self.candles_in_position,
            }

            if self.trades:
                self.trades[-1].update({
                    "exit": price, "pnl": pnl_pct,
                    "duration": self.candles_in_position
                })

            self.position_side        = None
            self.position_entry_price = 0.0
            self.candles_in_position  = 0
            self.candles_since_close  = 0

        return info

    def _observe(self) -> list[float]:
        """Buduje wektor obserwacji dla bieżącego kroku."""
        if self.step_idx >= len(self.candles):
            return [0.0] * self.fb.cfg.feature_count()

        candles_window = self.candles[max(0, self.step_idx - 99): self.step_idx + 1]
        candle         = self.candles[self.step_idx]

        closes  = [c.close  for c in candles_window]
        volumes = [c.volume for c in candles_window]
        opens   = [c.open   for c in candles_window]

        # Oblicz wskaźniki
        from trend_analyzer.indicators import ema_latest, rsi as calc_rsi, macd as calc_macd
        from trend_analyzer.indicators import market_structure, volume_analysis

        ema9   = ema_latest(closes, 9)  or closes[-1]
        ema21  = ema_latest(closes, 21) or closes[-1]
        ema50  = ema_latest(closes, 50) or closes[-1]
        rsi5m  = calc_rsi(closes) or 50.0
        macd_r = calc_macd(closes)
        struct = market_structure(candles_window)
        vol    = volume_analysis(candles_window)

        # Buy/Sell volume (szacowanie z świec)
        bull_vol = sum(c.volume for c in candles_window[-20:] if c.close >= c.open)
        bear_vol = sum(c.volume for c in candles_window[-20:] if c.close <  c.open)
        total_vol = bull_vol + bear_vol or 1.0
        self.cumulative_cvd += (bull_vol - bear_vol) / total_vol

        avg_candle = sum(abs(c.close - c.open) for c in candles_window[-20:]) / 20

        ext = self._get_external_signals()

        ms = {
            "close":        candle.close,
            "open":         candle.open,
            "high":         candle.high,
            "low":          candle.low,
            "volume":       candle.volume,
            "ema_9":        ema9,
            "ema_21":       ema21,
            "ema_50":       ema50,
            "rsi_1m":       rsi5m,
            "rsi_5m":       rsi5m,
            "rsi_15m":      ext.get("rsi_15m", rsi5m),
            "macd_hist":    macd_r[2] if macd_r else 0.0,
            "macd_line":    macd_r[0] if macd_r else 0.0,
            "macd_signal":  macd_r[1] if macd_r else 0.0,
            "higher_highs": struct["higher_highs"],
            "higher_lows":  struct.get("higher_lows", False),
            "lower_lows":   struct["lower_lows"],
            "lower_highs":  struct.get("lower_highs", False),
            "last_high":    struct["last_high"],
            "last_low":     struct["last_low"],
            "volume_ratio": vol["ratio"],
            "volume_trend": vol["trend"],
            "buy_volume":   bull_vol / total_vol,
            "sell_volume":  bear_vol / total_vol,
            "cvd":          max(-1.0, min(1.0, self.cumulative_cvd / 100)),
            "avg_candle_size": avg_candle,
            "position_side":      self.position_side,
            "position_pnl":       self._calc_pnl(candle.close),
            "candles_in_position": self.candles_in_position,
            "candles_since_close": self.candles_since_close,
            **ext,
        }

        return self.fb.build(ms)

    def _get_external_signals(self) -> dict:
        """Pobiera sygnały z Algo #1 i #2 dla bieżącego kroku."""
        if not self.extra_signals:
            return {
                "sentiment_score":    0.0,
                "trend_score_5m":     0.0,
                "trend_score_15m":    0.0,
                "tf_alignment":       0.5,
                "divergence_score":   0.0,
                "divergence_bullish": False,
                "divergence_bearish": False,
                "divergence_strength": "none",
                "rsi_15m":            50.0,
            }
        idx = min(self.step_idx, len(self.extra_signals) - 1)
        return self.extra_signals.get(idx, {})

    def _calc_pnl(self, current_price: float) -> float:
        """Oblicza % P&L aktualnej pozycji."""
        if not self.position_side or self.position_entry_price == 0:
            return 0.0
        if self.position_side == "long":
            return (current_price - self.position_entry_price) / self.position_entry_price
        else:
            return (self.position_entry_price - current_price) / self.position_entry_price

    def _check_missed_signal(self, action: str, ext: dict) -> bool:
        """Sprawdza czy agent pominął silny sygnał."""
        if self.position_side is not None:
            return False
        trend   = ext.get("trend_score_15m", 0.0)
        div     = ext.get("divergence_score", 0.0)
        div_str = ext.get("divergence_strength", "none")
        strong  = abs(trend) > 0.5 and div_str in ("medium", "strong") and abs(div) > 0.3
        return strong and action == "hold"

    def get_episode_stats(self) -> dict:
        """Zwraca statystyki zakończonego epizodu."""
        closed = [t for t in self.trades if "pnl" in t]
        wins   = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]

        total_return = (self.balance - self.initial_balance) / self.initial_balance

        max_dd = 0.0
        peak   = self.initial_balance
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        return {
            "total_return_pct":  round(total_return * 100, 2),
            "final_balance":     round(self.balance, 2),
            "total_trades":      len(closed),
            "win_rate":          round(len(wins) / max(len(closed), 1) * 100, 1),
            "avg_win_pct":       round(sum(t["pnl"] for t in wins)   / max(len(wins), 1) * 100, 3),
            "avg_loss_pct":      round(sum(t["pnl"] for t in losses) / max(len(losses), 1) * 100, 3),
            "total_reward":      round(sum(self.rewards), 4),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "sharpe":            self._calc_sharpe(),
        }

    def _calc_sharpe(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        arr  = self.rewards
        mean = sum(arr) / len(arr)
        std  = math.sqrt(sum((x - mean)**2 for x in arr) / len(arr))
        return round(mean / std * math.sqrt(252) if std > 0 else 0.0, 3)