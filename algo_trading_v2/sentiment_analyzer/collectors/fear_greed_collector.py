"""
collectors/fear_greed_collector.py
------------------------------------
Fear & Greed Index z alternative.me
Darmowy, bez klucza API, aktualizowany raz dziennie.
Zakres: 0 (Extreme Fear) → 100 (Extreme Greed)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from sentiment_analyzer.output_schema import SourceSignal

logger = logging.getLogger(__name__)

API_URL         = "https://api.alternative.me/fng/"
CACHE_TTL_SEC   = 3600
REQUEST_TIMEOUT = 10


class FearGreedCollector:

    def __init__(self):
        self._cache:           Optional[dict] = None
        self._cache_timestamp: float = 0.0

    def get_signal(self, symbol: str = "BTC") -> Optional[SourceSignal]:
        raw = self._fetch()
        if raw is None:
            return None

        score      = round((raw["value"] - 50) / 50, 4)   # 0-100 → -1..+1
        confidence = round(0.4 + (abs(score) * 0.5), 4)   # ekstrema = wyższa pewność

        return SourceSignal(
            source_name  = "fear_greed",
            score        = score,
            confidence   = min(confidence, 0.95),
            sample_count = 1,
            timestamp    = datetime.fromtimestamp(raw["timestamp"], tz=timezone.utc),
            is_stale     = (time.time() - raw["timestamp"]) > 108000,  # 30h
            raw_metadata = {
                "value":          raw["value"],
                "classification": raw["classification"],
            }
        )

    def _fetch(self) -> Optional[dict]:
        now = time.time()
        if self._cache and (now - self._cache_timestamp) < CACHE_TTL_SEC:
            return self._cache

        try:
            r = requests.get(API_URL, params={"limit": 1}, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            entry = r.json()["data"][0]
            self._cache = {
                "value":          int(entry["value"]),
                "classification": entry["value_classification"],
                "timestamp":      int(entry["timestamp"]),
            }
            self._cache_timestamp = now
            logger.info(f"FearGreed: {self._cache['value']} ({self._cache['classification']})")
            return self._cache

        except Exception as e:
            logger.error(f"FearGreed: błąd — {e}")
            return self._cache  # stary cache lepszy niż nic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = FearGreedCollector().get_signal()
    if s:
        print(f"Score: {s.score:+.4f}  Conf: {s.confidence:.4f}  "
              f"Value: {s.raw_metadata['value']} ({s.raw_metadata['classification']})")
