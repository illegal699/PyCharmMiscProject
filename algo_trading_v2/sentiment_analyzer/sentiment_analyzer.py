"""
sentiment_analyzer.py
-----------------------
Algorytm #1 — SentimentAnalyzer.
Źródła: Fear & Greed Index + Google Trends (oba darmowe, bez klucza).
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from sentiment_analyzer.collectors.fear_greed_collector import FearGreedCollector
from sentiment_analyzer.collectors.trends_collector     import TrendsCollector
from sentiment_analyzer.output_schema                   import SentimentSignal
from sentiment_analyzer.processors.signal_normalizer    import SignalNormalizer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Algorytm #1 — analizator sentymentu rynkowego.

    Interfejs dla Algo #2 i #3:
        signal = analyzer.get_latest_signal()    # ostatni sygnał (nie blokuje)
        signal = analyzer.get_signal_sync()      # nowy sygnał (blokuje)
        signal = await analyzer.get_signal_async()
    """

    def __init__(self, symbol: str = "BTC/USDT", poll_interval: int = 60):
        self.symbol        = symbol
        self.poll_interval = poll_interval
        self._btc          = symbol.split("/")[0]   # "BTC"

        self._fear_greed  = FearGreedCollector()
        self._trends      = TrendsCollector()
        self._normalizer  = SignalNormalizer(symbol=symbol)
        self._executor    = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentiment")

        self._latest_signal: Optional[SentimentSignal] = None
        self._last_update:   float = 0.0
        self._running = False

        logger.info(f"SentimentAnalyzer: gotowy | symbol={symbol}")

    # ------------------------------------------------------------------
    # Publiczny interfejs
    # ------------------------------------------------------------------

    def get_latest_signal(self) -> Optional[SentimentSignal]:
        """Zwraca ostatni sygnał — nie blokuje, używaj w Algo #2 i #3."""
        return self._latest_signal

    def get_signal_sync(self) -> Optional[SentimentSignal]:
        """Pobiera nowy sygnał synchronicznie."""
        return self._pipeline()

    async def get_signal_async(self) -> Optional[SentimentSignal]:
        """Pobiera nowy sygnał asynchronicznie — nie blokuje event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._pipeline)

    def update_weights(self, weights: dict) -> None:
        """Aktualizuje wagi źródeł — wywoływane przez Algorytm #4."""
        self._normalizer.update_weights(weights)

    async def start_polling(self) -> None:
        """Ciągłe pobieranie sygnałów w tle."""
        self._running = True
        logger.info(f"SentimentAnalyzer: start polling co {self.poll_interval}s")

        while self._running:
            try:
                signal = await self.get_signal_async()
                if signal:
                    self._latest_signal = signal
                    self._last_update   = time.time()
                    self._log(signal)
            except Exception as e:
                logger.error(f"SentimentAnalyzer: błąd pętli — {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        logger.info("SentimentAnalyzer: zatrzymany")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _pipeline(self) -> Optional[SentimentSignal]:
        """Pobiera sygnały równolegle i składa w kompozyt."""
        f_fg = self._executor.submit(self._safe, self._fear_greed, self._btc)
        f_tr = self._executor.submit(self._safe, self._trends,     self._btc)

        fg = f_fg.result(timeout=15)
        tr = f_tr.result(timeout=30)   # Trends może być wolniejszy

        return self._normalizer.compose(
            fear_greed_signal = fg,
            trends_signal     = tr,
        )

    def _safe(self, collector, symbol: str):
        try:
            return collector.get_signal(symbol)
        except Exception as e:
            logger.error(f"SentimentAnalyzer: błąd {collector.__class__.__name__} — {e}")
            return None

    def _log(self, s: SentimentSignal) -> None:
        emoji = {"strongly_bullish": "🟢🟢", "bullish": "🟢",
                 "neutral": "⚪", "bearish": "🔴", "strongly_bearish": "🔴🔴"}
        e = emoji.get(s.direction.value, "❓")
        logger.info(
            f"[SENTIMENT] {e} score={s.composite_score:+.3f}  "
            f"conf={s.confidence:.2f}  quality={s.quality.value}  "
            f"vel={s.sentiment_velocity:+.3f}  session={s.market_session}  "
            f"tradeable={s.is_tradeable}"
        )
        if s.anomaly_detected:
            logger.warning(f"[SENTIMENT] ⚠️  Anomalia: {s.anomaly_source}")


# ---------------------------------------------------------------------------
# Test manualny
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        print("\n" + "="*55)
        print("  SentimentAnalyzer — test")
        print("="*55 + "\n")

        analyzer = SentimentAnalyzer()
        print("Pobieranie sygnału (może chwilę zająć)...\n")

        signal = await analyzer.get_signal_async()

        if signal:
            fg = signal.fear_greed_signal
            tr = signal.trends_signal

            print(f"✅ Sygnał:")
            print(f"   Score:      {signal.composite_score:+.4f}")
            print(f"   Direction:  {signal.direction.value}")
            print(f"   Confidence: {signal.confidence:.4f}")
            print(f"   Quality:    {signal.quality.value}")
            print(f"   Velocity:   {signal.sentiment_velocity:+.4f}")
            print(f"   Session:    {signal.market_session}")
            print(f"   Tradeable:  {signal.is_tradeable}")
            print(f"\n   Składowe:")
            if fg:
                print(f"   Fear&Greed  score={fg.score:+.4f}  conf={fg.confidence:.4f}  "
                      f"value={fg.raw_metadata.get('value')} ({fg.raw_metadata.get('classification')})")
            if tr:
                print(f"   Trends      score={tr.score:+.4f}  conf={tr.confidence:.4f}  "
                      f"current={tr.raw_metadata.get('current')}  "
                      f"avg30d={tr.raw_metadata.get('avg_30d')}")
        else:
            print("❌ Brak sygnału")

        analyzer.stop()

    asyncio.run(main())
