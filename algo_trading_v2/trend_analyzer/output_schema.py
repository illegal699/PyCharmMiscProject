"""
trend_analyzer/output_schema.py
---------------------------------
Kontrakt danych Algorytmu #2 — TrendAnalyzer.
Konsumowany przez: Algorytm #3 (trader), Algorytm #4 (meta).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TrendDirection(Enum):
    STRONG_UP   = "strong_up"    # wyraźny trend wzrostowy
    UP          = "up"           # trend wzrostowy
    BALANCE     = "balance"      # rynek w równowadze / konsolidacja
    DOWN        = "down"         # trend spadkowy
    STRONG_DOWN = "strong_down"  # wyraźny trend spadkowy


class TrendStrength(Enum):
    STRONG = "strong"   # > 0.70
    MEDIUM = "medium"   # 0.40 .. 0.70
    WEAK   = "weak"     # < 0.40


class MarketPhase(Enum):
    TRENDING     = "trending"      # wyraźny kierunek
    CONSOLIDATION = "consolidation" # wąski zakres, brak kierunku
    BREAKOUT     = "breakout"      # właśnie wyszedł z konsolidacji
    REVERSAL     = "reversal"      # możliwe odwrócenie trendu


@dataclass
class TimeframeAnalysis:
    """Analiza pojedynczego timeframe'u."""
    timeframe:   str              # "1m" | "5m" | "15m"

    # Trend
    direction:   TrendDirection
    strength:    float            # 0.0 .. 1.0
    score:       float            # -1.0 .. +1.0 (ujemny = down, dodatni = up)

    # Wskaźniki
    ema_fast:    float            # EMA 9
    ema_slow:    float            # EMA 21
    ema_trend:   float            # EMA 50
    rsi:         float            # 0 .. 100
    macd:        float            # wartość MACD
    macd_signal: float            # linia sygnału
    macd_hist:   float            # histogram

    # Struktura rynku
    higher_highs: bool            # HH — bullish struktura
    lower_lows:   bool            # LL — bearish struktura
    last_high:    float           # ostatni swing high
    last_low:     float           # ostatni swing low

    # Wolumen
    volume_trend:  str            # "increasing" | "decreasing" | "neutral"
    volume_ratio:  float          # obecny wolumen / średni wolumen

    # Faza
    phase:        MarketPhase
    price:        float           # aktualna cena zamknięcia
    timestamp:    datetime


@dataclass
class TrendSignal:
    """
    Główny output Algorytmu #2.
    Konsumowany przez: Algorytm #3 (trader), Algorytm #4 (meta).
    """

    timestamp:  datetime
    symbol:     str
    signal_id:  str

    # --- Główny kierunek (z hierarchii TF) ---
    primary_direction: TrendDirection    # z 15m — nadrzędny kierunek
    entry_direction:   TrendDirection    # z 5m — kierunek wejścia
    score:             float             # -1.0 .. +1.0
    strength:          TrendStrength
    confidence:        float             # 0.0 .. 1.0

    # --- Analiza per timeframe ---
    tf_15m: TimeframeAnalysis
    tf_5m:  TimeframeAnalysis
    tf_1m:  TimeframeAnalysis

    # --- Zgodność timeframe'ów ---
    tf_alignment:      float    # 0.0 = sprzeczne, 1.0 = wszystkie zgodne
    tf_alignment_desc: str      # np. "15m↑ 5m↑ 1m↓"

    # --- Sygnały dla Algo #3 ---
    is_tradeable:       bool
    skip_reason:        Optional[str] = None
    suggested_side:     Optional[str] = None   # "long" | "short" | None
    invalidation_level: Optional[float] = None  # poziom unieważnienia sygnału

    # --- Dywergencje ---
    divergences_5m:  Optional[object] = None   # DivergenceResult
    divergences_15m: Optional[object] = None   # DivergenceResult
    divergence_score: float = 0.0              # łączny score dywergencji

    # --- Kontekst ---
    market_phase:   MarketPhase = MarketPhase.TRENDING
    current_price:  float = 0.0
    score_history:  list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp":         self.timestamp.isoformat(),
            "symbol":            self.symbol,
            "signal_id":         self.signal_id,
            "primary_direction": self.primary_direction.value,
            "entry_direction":   self.entry_direction.value,
            "score":             round(self.score, 4),
            "strength":          self.strength.value,
            "confidence":        round(self.confidence, 4),
            "tf_alignment":      round(self.tf_alignment, 4),
            "tf_alignment_desc": self.tf_alignment_desc,
            "is_tradeable":      self.is_tradeable,
            "skip_reason":       self.skip_reason,
            "suggested_side":    self.suggested_side,
            "current_price":     self.current_price,
            "market_phase":      self.market_phase.value,
        }


def classify_trend_strength(strength: float) -> TrendStrength:
    if strength > 0.70: return TrendStrength.STRONG
    if strength > 0.40: return TrendStrength.MEDIUM
    return TrendStrength.WEAK


def score_to_direction(score: float) -> TrendDirection:
    if score >  0.60: return TrendDirection.STRONG_UP
    if score >  0.20: return TrendDirection.UP
    if score < -0.60: return TrendDirection.STRONG_DOWN
    if score < -0.20: return TrendDirection.DOWN
    return TrendDirection.BALANCE
