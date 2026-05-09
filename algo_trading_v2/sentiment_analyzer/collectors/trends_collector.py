"""
collectors/trends_collector.py
--------------------------------
Google Trends jako sygnał sentymentu przez pytrends.
Darmowy, bez klucza API.

Instalacja: pip install pytrends

Logika:
- Wysoki trend wyszukiwań BTC = wzrost zainteresowania = często bullish
- Bardzo gwałtowny spike = możliwy szczyt (FOMO) = kontrariańsko bearish
- Niskie zainteresowanie = akumulacja lub brak zainteresowania
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from sentiment_analyzer.output_schema import SourceSignal

logger = logging.getLogger(__name__)

CACHE_TTL_SEC = 1800    # 30 minut (Trends nie zmienia się co minutę)


class TrendsCollector:

    def __init__(self):
        self._cache:           Optional[dict] = None
        self._cache_timestamp: float = 0.0
        self._pytrends = None

    def _ensure_pytrends(self):
        if self._pytrends is not None:
            return True
        try:
            from pytrends.request import TrendReq
            self._pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            return True
        except ImportError:
            logger.error("TrendsCollector: zainstaluj pytrends: pip install pytrends")
            return False

    def get_signal(self, symbol: str = "BTC") -> Optional[SourceSignal]:
        data = self._fetch(symbol)
        if data is None:
            return None

        current  = data["current"]    # 0-100 (ostatni punkt)
        avg_7d   = data["avg_7d"]     # średnia z 7 dni
        avg_30d  = data["avg_30d"]    # średnia z 30 dni

        # Sygnał względny: czy obecny trend jest powyżej/poniżej średniej
        if avg_30d > 0:
            relative_vs_30d = (current - avg_30d) / max(avg_30d, 1)
        else:
            relative_vs_30d = 0.0

        # Normalizacja do [-1, +1]
        # +1 = obecny trend 2x powyżej 30d średniej (silne zainteresowanie)
        # -1 = obecny trend 2x poniżej 30d średniej (brak zainteresowania)
        score = max(-1.0, min(1.0, relative_vs_30d))

        # Korekta kontrariańska: bardzo wysoki spike (>90) często = szczyt FOMO
        if current > 90 and score > 0.5:
            score *= 0.6    # przytnij euforię
            logger.info(f"Trends: spike kontrariański (current={current}), koreguję score")

        # Pewność: wyższa gdy dane są stabilne (mała wariancja)
        variance    = data.get("variance", 20)
        confidence  = round(max(0.35, 0.75 - (variance / 200)), 4)

        return SourceSignal(
            source_name  = "trends",
            score        = round(score, 4),
            confidence   = confidence,
            sample_count = data["points_count"],
            timestamp    = datetime.now(tz=timezone.utc),
            is_stale     = self._is_stale(),
            raw_metadata = {
                "current":  current,
                "avg_7d":   round(avg_7d, 1),
                "avg_30d":  round(avg_30d, 1),
                "keyword":  data["keyword"],
            }
        )

    def _fetch(self, symbol: str) -> Optional[dict]:
        now = time.time()
        if self._cache and (now - self._cache_timestamp) < CACHE_TTL_SEC:
            logger.debug("Trends: cache hit")
            return self._cache

        if not self._ensure_pytrends():
            return None

        # Słowa kluczowe dla danego symbolu
        keyword_map = {"BTC": "Bitcoin", "ETH": "Ethereum"}
        keyword     = keyword_map.get(symbol, "Bitcoin")

        try:
            self._pytrends.build_payload(
                kw_list    = [keyword],
                timeframe  = "today 1-m",   # ostatnie 30 dni
                geo        = "",             # globalnie
            )
            df = self._pytrends.interest_over_time()

            if df is None or df.empty:
                logger.warning("Trends: brak danych")
                return None

            values = df[keyword].tolist()
            if not values:
                return None

            current     = float(values[-1])
            avg_7d      = float(sum(values[-7:])  / min(len(values), 7))
            avg_30d     = float(sum(values)        / len(values))
            variance    = float(max(values) - min(values))

            self._cache = {
                "current":      current,
                "avg_7d":       avg_7d,
                "avg_30d":      avg_30d,
                "variance":     variance,
                "points_count": len(values),
                "keyword":      keyword,
            }
            self._cache_timestamp = now
            logger.info(f"Trends: {keyword} current={current:.0f}  avg30d={avg_30d:.1f}")
            return self._cache

        except Exception as e:
            logger.error(f"Trends: błąd — {e}")
            return self._cache

    def _is_stale(self) -> bool:
        if self._cache_timestamp == 0:
            return True
        return (time.time() - self._cache_timestamp) > (CACHE_TTL_SEC * 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = TrendsCollector().get_signal("BTC")
    if s:
        m = s.raw_metadata
        print(f"Score:   {s.score:+.4f}")
        print(f"Conf:    {s.confidence:.4f}")
        print(f"Current: {m['current']}  Avg7d: {m['avg_7d']}  Avg30d: {m['avg_30d']}")
    else:
        print("Brak danych")
