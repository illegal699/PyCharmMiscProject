"""
trend_analyzer/data_fetcher.py
--------------------------------
Pobiera dane OHLCV z Binance API.
Używa klucza publicznego (do odczytu danych nie potrzeba podpisywania).

Env vars:
    BINANCE_API_KEY    — opcjonalny dla danych publicznych
    BINANCE_SECRET_KEY — opcjonalny dla danych publicznych
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL        = "https://api.binance.com"
REQUEST_TIMEOUT = 10

# Cache TTL per timeframe
CACHE_TTL = {
    "1m":  30,    # 30 sekund
    "5m":  60,    # 1 minuta
    "15m": 120,   # 2 minuty
}

# Ile świec pobierać per timeframe
CANDLES_COUNT = {
    "1m":  100,
    "5m":  100,
    "15m": 100,
}


class Candle:
    """Pojedyncza świeca OHLCV."""
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, raw: list):
        self.timestamp = int(raw[0])
        self.open      = float(raw[1])
        self.high      = float(raw[2])
        self.low       = float(raw[3])
        self.close     = float(raw[4])
        self.volume    = float(raw[5])


class BinanceDataFetcher:
    """
    Pobiera dane OHLCV z Binance.
    Cache per timeframe — nie wykonuje requestu częściej niż TTL.
    """

    def __init__(self):
        self._api_key = os.getenv("BINANCE_API_KEY", "")
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["X-MBX-APIKEY"] = self._api_key

        # Cache: {symbol_tf: {"candles": [...], "ts": float}}
        self._cache: dict = {}

    def get_candles(
        self,
        symbol:    str = "BTCUSDT",
        timeframe: str = "5m",
        limit:     int = None,
    ) -> Optional[list[Candle]]:
        """
        Zwraca listę świec dla danego symbolu i timeframe'u.
        Najnowsza świeca jest ostatnia na liście.
        """
        limit    = limit or CANDLES_COUNT.get(timeframe, 100)
        cache_key = f"{symbol}_{timeframe}"
        ttl       = CACHE_TTL.get(timeframe, 60)
        now       = time.time()

        cached = self._cache.get(cache_key)
        if cached and (now - cached["ts"]) < ttl:
            logger.debug(f"Binance: cache hit {cache_key}")
            return cached["candles"]

        candles = self._fetch(symbol, timeframe, limit)
        if candles:
            self._cache[cache_key] = {"candles": candles, "ts": now}

        return candles

    def get_latest_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Zwraca aktualną cenę symbolu."""
        try:
            r = self._session.get(
                f"{BASE_URL}/api/v3/ticker/price",
                params={"symbol": symbol},
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception as e:
            logger.error(f"Binance: błąd ceny — {e}")
            return None

    def _fetch(self, symbol: str, timeframe: str, limit: int) -> Optional[list[Candle]]:
        """Pobiera świece z Binance REST API."""
        try:
            r = self._session.get(
                f"{BASE_URL}/api/v3/klines",
                params={
                    "symbol":   symbol,
                    "interval": timeframe,
                    "limit":    limit,
                },
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            candles = [Candle(row) for row in r.json()]
            logger.debug(f"Binance: fetched {len(candles)} świec {symbol} {timeframe}")
            return candles

        except Exception as e:
            logger.error(f"Binance: błąd fetch {symbol} {timeframe} — {e}")
            cached = self._cache.get(f"{symbol}_{timeframe}")
            return cached["candles"] if cached else None
