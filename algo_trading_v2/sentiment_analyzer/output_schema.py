"""
output_schema.py
----------------
Kontrakt danych między algorytmami systemu algo-tradingowego.
Źródła: Fear & Greed Index + Google Trends
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SentimentDirection(Enum):
    STRONGLY_BULLISH = "strongly_bullish"   # score > 0.6
    BULLISH          = "bullish"            # score > 0.2
    NEUTRAL          = "neutral"            # score -0.2 .. 0.2
    BEARISH          = "bearish"            # score < -0.2
    STRONGLY_BEARISH = "strongly_bearish"   # score < -0.6


class SignalQuality(Enum):
    HIGH   = "high"     # confidence > 0.75
    MEDIUM = "medium"   # confidence 0.4 .. 0.75
    LOW    = "low"      # confidence < 0.4


@dataclass
class SourceSignal:
    """Sygnał z pojedynczego źródła danych."""
    source_name:  str
    score:        float        # -1.0 .. +1.0
    confidence:   float        # 0.0 .. 1.0
    sample_count: int
    timestamp:    datetime
    is_stale:     bool = False
    raw_metadata: dict = field(default_factory=dict)


@dataclass
class SentimentSignal:
    """
    Główny output Algorytmu #1.
    Konsumowany przez: Algorytm #2, #3, #4.
    """

    # Identyfikacja
    timestamp:  datetime
    symbol:     str
    signal_id:  str

    # Główny sygnał
    composite_score: float
    confidence:      float
    direction:       SentimentDirection
    quality:         SignalQuality

    # Składowe
    fear_greed_signal: Optional[SourceSignal]
    trends_signal:     Optional[SourceSignal]

    # Wagi
    weights_used: dict = field(default_factory=lambda: {
        "fear_greed": 0.50,
        "trends":     0.50,
    })

    # Dynamika
    sentiment_velocity:     float = 0.0
    sentiment_acceleration: float = 0.0
    anomaly_detected:       bool  = False
    anomaly_source:         Optional[str] = None

    # Relevancja TF
    tf_1m_relevance:  float = 0.0
    tf_5m_relevance:  float = 0.0
    tf_15m_relevance: float = 0.0

    # Kontekst
    market_session:  str = "unknown"
    dominant_source: str = "unknown"

    # Flagi
    is_tradeable: bool          = True
    skip_reason:  Optional[str] = None

    # Historia
    score_history: list = field(default_factory=list)

    def get_tf_relevance(self, timeframe: str) -> float:
        return {"1m": self.tf_1m_relevance,
                "5m": self.tf_5m_relevance,
                "15m": self.tf_15m_relevance}.get(timeframe, 0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp.isoformat(),
            "symbol":          self.symbol,
            "signal_id":       self.signal_id,
            "composite_score": round(self.composite_score, 4),
            "confidence":      round(self.confidence, 4),
            "direction":       self.direction.value,
            "quality":         self.quality.value,
            "velocity":        round(self.sentiment_velocity, 4),
            "anomaly":         self.anomaly_detected,
            "anomaly_source":  self.anomaly_source,
            "tf_1m":           round(self.tf_1m_relevance, 4),
            "tf_5m":           round(self.tf_5m_relevance, 4),
            "tf_15m":          round(self.tf_15m_relevance, 4),
            "session":         self.market_session,
            "dominant":        self.dominant_source,
            "is_tradeable":    self.is_tradeable,
            "skip_reason":     self.skip_reason,
            "sources": {
                "fear_greed": self._src(self.fear_greed_signal),
                "trends":     self._src(self.trends_signal),
            }
        }

    @staticmethod
    def _src(s: Optional[SourceSignal]) -> Optional[dict]:
        if s is None:
            return None
        return {
            "score":      round(s.score, 4),
            "confidence": round(s.confidence, 4),
            "n":          s.sample_count,
            "stale":      s.is_stale,
            "meta":       s.raw_metadata,
        }


def classify_direction(score: float) -> SentimentDirection:
    if score > 0.6:   return SentimentDirection.STRONGLY_BULLISH
    if score > 0.2:   return SentimentDirection.BULLISH
    if score < -0.6:  return SentimentDirection.STRONGLY_BEARISH
    if score < -0.2:  return SentimentDirection.BEARISH
    return SentimentDirection.NEUTRAL


def classify_quality(confidence: float) -> SignalQuality:
    if confidence > 0.75: return SignalQuality.HIGH
    if confidence > 0.40: return SignalQuality.MEDIUM
    return SignalQuality.LOW
