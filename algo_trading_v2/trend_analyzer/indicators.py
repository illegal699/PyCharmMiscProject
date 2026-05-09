"""
trend_analyzer/indicators.py
------------------------------
Obliczanie wskaźników technicznych bez zewnętrznych bibliotek.
Tylko Python + math — zero zależności.

Wskaźniki:
- EMA (9, 21, 50)
- RSI (14)
- MACD (12, 26, 9)
- Struktura rynku (HH/HL/LH/LL)
- Wolumen relative
"""

import math
from typing import Optional
from trend_analyzer.data_fetcher import Candle


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return []

    k      = 2 / (period + 1)
    result = [sum(values[:period]) / period]   # SMA jako seed

    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))

    return result


def ema_latest(values: list[float], period: int) -> Optional[float]:
    """Zwraca tylko ostatnią wartość EMA."""
    e = ema(values, period)
    return e[-1] if e else None


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return None

    gains  = []
    losses = []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # Pierwsze avg (SMA)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smoothed (Wilder)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[tuple[float, float, float]]:
    """
    MACD.
    Zwraca (macd_line, signal_line, histogram) lub None.
    """
    if len(closes) < slow + signal:
        return None

    ema_fast_vals = ema(closes, fast)
    ema_slow_vals = ema(closes, slow)

    # Wyrównaj długości
    diff = len(ema_fast_vals) - len(ema_slow_vals)
    ema_fast_aligned = ema_fast_vals[diff:]

    macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow_vals)]

    if len(macd_line) < signal:
        return None

    signal_line_vals = ema(macd_line, signal)
    if not signal_line_vals:
        return None

    macd_val   = macd_line[-1]
    signal_val = signal_line_vals[-1]
    hist_val   = macd_val - signal_val

    return round(macd_val, 6), round(signal_val, 6), round(hist_val, 6)


# ---------------------------------------------------------------------------
# Struktura rynku
# ---------------------------------------------------------------------------

def market_structure(
    candles: list[Candle],
    lookback: int = 20,
) -> dict:
    """
    Analiza struktury rynku: Higher Highs, Higher Lows, Lower Highs, Lower Lows.
    Zwraca słownik z wynikami analizy.
    """
    if len(candles) < lookback:
        lookback = len(candles)

    recent = candles[-lookback:]
    highs  = [c.high  for c in recent]
    lows   = [c.low   for c in recent]

    # Podziel na dwie połowy i porównaj
    mid    = len(recent) // 2
    first_highs = highs[:mid]
    second_highs = highs[mid:]
    first_lows  = lows[:mid]
    second_lows = lows[mid:]

    avg_first_high  = sum(first_highs)  / len(first_highs)
    avg_second_high = sum(second_highs) / len(second_highs)
    avg_first_low   = sum(first_lows)   / len(first_lows)
    avg_second_low  = sum(second_lows)  / len(second_lows)

    higher_highs = avg_second_high > avg_first_high
    higher_lows  = avg_second_low  > avg_first_low
    lower_highs  = avg_second_high < avg_first_high
    lower_lows   = avg_second_low  < avg_first_low

    # Swing points
    last_high = max(highs[-10:]) if len(highs) >= 10 else max(highs)
    last_low  = min(lows[-10:])  if len(lows)  >= 10 else min(lows)

    # Bullish: HH + HL | Bearish: LH + LL
    bullish_structure = higher_highs and higher_lows
    bearish_structure = lower_highs  and lower_lows

    return {
        "higher_highs":       higher_highs,
        "higher_lows":        higher_lows,
        "lower_highs":        lower_highs,
        "lower_lows":         lower_lows,
        "bullish_structure":  bullish_structure,
        "bearish_structure":  bearish_structure,
        "last_high":          last_high,
        "last_low":           last_low,
    }


# ---------------------------------------------------------------------------
# Wolumen
# ---------------------------------------------------------------------------

def volume_analysis(candles: list[Candle], period: int = 20) -> dict:
    """
    Analiza wolumenu.
    Zwraca trend wolumenu i stosunek do średniej.
    """
    if len(candles) < 2:
        return {"trend": "neutral", "ratio": 1.0}

    volumes = [c.volume for c in candles]
    current_vol = volumes[-1]

    # Średnia wolumenu
    avg_vol = sum(volumes[-period:]) / min(len(volumes), period)
    ratio   = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Trend wolumenu (ostatnie 5 świec)
    recent_vols = volumes[-5:]
    if len(recent_vols) >= 3:
        first_half = sum(recent_vols[:len(recent_vols)//2])
        second_half = sum(recent_vols[len(recent_vols)//2:])
        if second_half > first_half * 1.1:
            trend = "increasing"
        elif second_half < first_half * 0.9:
            trend = "decreasing"
        else:
            trend = "neutral"
    else:
        trend = "neutral"

    return {
        "trend":   trend,
        "ratio":   round(ratio, 3),
        "current": current_vol,
        "average": round(avg_vol, 2),
    }
