"""
processors/signal_normalizer.py
---------------------------------
Składa sygnały z Fear & Greed i Google Trends w jeden SentimentSignal.
"""

import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from sentiment_analyzer.output_schema import (
    SentimentSignal, SourceSignal,
    classify_direction, classify_quality,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "fear_greed": 0.50,
    "trends":     0.50,
}

# Jak szybko każde źródło reaguje na rynek (relevancja dla TF)
TF_PROFILE = {
    #               1m    5m    15m
    "fear_greed": (0.25, 0.40, 0.60),
    "trends":     (0.20, 0.35, 0.55),
}

HISTORY_SIZE      = 20
ANOMALY_THRESHOLD = 0.40


class SignalNormalizer:

    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol   = symbol
        self._history = deque(maxlen=HISTORY_SIZE)
        self._weights = DEFAULT_WEIGHTS.copy()

    def update_weights(self, new_weights: dict) -> None:
        total = sum(new_weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in new_weights.items()}

    def compose(
        self,
        fear_greed_signal: Optional[SourceSignal] = None,
        trends_signal:     Optional[SourceSignal] = None,
    ) -> Optional[SentimentSignal]:

        sources = {
            "fear_greed": fear_greed_signal,
            "trends":     trends_signal,
        }
        active = {n: s for n, s in sources.items() if s is not None and not s.is_stale}

        if not active:
            logger.warning("SignalNormalizer: brak aktywnych sygnałów")
            return None

        # Kompozyt ważony
        composite, confidence, dominant = self._composite(active)

        # Dynamika
        velocity, acceleration = self._velocity(composite)

        # Anomalia
        anomaly, anomaly_src = self._anomaly(composite, active)

        # TF relevancja
        tf1, tf5, tf15 = self._tf_relevance(active)

        # Sesja
        session = self._session()

        # Czy tradeable
        tradeable, skip = self._tradeable(confidence, anomaly)

        self._history.append(composite)

        return SentimentSignal(
            timestamp               = datetime.now(tz=timezone.utc),
            symbol                  = self.symbol,
            signal_id               = str(uuid.uuid4())[:8],
            composite_score         = round(composite, 4),
            confidence              = round(confidence, 4),
            direction               = classify_direction(composite),
            quality                 = classify_quality(confidence),
            fear_greed_signal       = fear_greed_signal,
            trends_signal           = trends_signal,
            weights_used            = self._weights.copy(),
            sentiment_velocity      = round(velocity, 4),
            sentiment_acceleration  = round(acceleration, 4),
            anomaly_detected        = anomaly,
            anomaly_source          = anomaly_src,
            tf_1m_relevance         = round(tf1, 4),
            tf_5m_relevance         = round(tf5, 4),
            tf_15m_relevance        = round(tf15, 4),
            market_session          = session,
            dominant_source         = dominant,
            is_tradeable            = tradeable,
            skip_reason             = skip,
            score_history           = list(self._history)[-10:],
        )

    def _composite(self, active: dict) -> tuple[float, float, str]:
        w_score = w_conf = w_total = 0.0
        dominant = "unknown"
        max_contrib = 0.0

        for name, sig in active.items():
            w = self._weights.get(name, 0.0) * sig.confidence
            w_score  += sig.score      * w
            w_conf   += sig.confidence * self._weights.get(name, 0.0)
            w_total  += w
            contrib   = abs(sig.score) * w
            if contrib > max_contrib:
                max_contrib = contrib
                dominant = name

        if w_total == 0:
            return 0.0, 0.0, "none"

        composite  = w_score / w_total
        confidence = w_conf / sum(self._weights.get(n, 0) for n in active)
        bonus      = min((len(active) - 1) * 0.05, 0.10)
        return composite, min(confidence + bonus, 0.92), dominant

    def _velocity(self, current: float) -> tuple[float, float]:
        if len(self._history) < 2:
            return 0.0, 0.0
        velocity = current - self._history[-1]
        if len(self._history) < 3:
            return velocity, 0.0
        acceleration = velocity - (self._history[-1] - self._history[-2])
        return velocity, acceleration

    def _anomaly(self, composite: float, active: dict) -> tuple[bool, Optional[str]]:
        if self._history and abs(composite - self._history[-1]) > ANOMALY_THRESHOLD:
            return True, "composite_spike"
        for name, sig in active.items():
            if abs(sig.score) > 0.85:
                return True, f"{name}_extreme"
        return False, None

    def _tf_relevance(self, active: dict) -> tuple[float, float, float]:
        t1 = t5 = t15 = w_total = 0.0
        for name, sig in active.items():
            w = self._weights.get(name, 0.0) * sig.confidence
            p = TF_PROFILE.get(name, (0.3, 0.4, 0.5))
            t1 += p[0] * w; t5 += p[1] * w; t15 += p[2] * w
            w_total += w
        if w_total == 0:
            return 0.3, 0.4, 0.5
        return t1 / w_total, t5 / w_total, t15 / w_total

    def _tradeable(self, confidence: float, anomaly: bool) -> tuple[bool, Optional[str]]:
        if confidence < 0.30:
            return False, f"low_confidence({confidence:.2f})"
        if anomaly and confidence < 0.55:
            return False, "anomaly_low_confidence"
        return True, None

    def _session(self) -> str:
        h = datetime.now(tz=timezone.utc).hour
        if h < 8:    return "asian"
        if h < 13:   return "european"
        if h < 17:   return "us_overlap"
        if h < 22:   return "us"
        return "late_us"
