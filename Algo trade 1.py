"""
sentiment_analyzer.py
-----------------------
Główna klasa Algorytmu #1 — SentimentAnalyzer.
Spinaje wszystkie kolektory, NLP processor i normalizer w jeden pipeline.

Użycie:
    analyzer = SentimentAnalyzer(config)
    signal   = await analyzer.get_signal()   # lub analyzer.get_signal_sync()

Sygnał (SentimentSignal) jest gotowy do konsumpcji przez Algorytm #2 i #3.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from sentiment_analyzer.collectors.fear_greed_collector import FearGreedCollector
from sentiment_analyzer.collectors.news_collector       import NewsCollector
from sentiment_analyzer.collectors.onchain_collector    import OnchainCollector
from sentiment_analyzer.collectors.reddit_collector     import RedditCollector
from sentiment_analyzer.collectors.stocktwits_collector import StockTwitsCollector
from sentiment_analyzer.output_schema                   import SentimentSignal
from sentiment_analyzer.processors.nlp_processor        import NLPProcessor
from sentiment_analyzer.processors.signal_normalizer    import SignalNormalizer

logger = logging.getLogger(__name__)


class SentimentAnalyzerConfig:
    """Konfiguracja SentimentAnalyzer."""

    def __init__(self):
        # API keys — czytaj ze zmiennych środowiskowych
        self.reddit_client_id     = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.cryptopanic_api_key  = os.getenv("CRYPTOPANIC_API_KEY", "")
        self.glassnode_api_key    = os.getenv("GLASSNODE_API_KEY", "")

        # Symbol
        self.symbol     = os.getenv("TRADING_SYMBOL", "BTC/USDT")
        self.btc_symbol = "BTC"   # dla API które nie używają pary

        # NLP
        self.use_finbert = os.getenv("USE_FINBERT", "true").lower() == "true"
        self.nlp_device  = os.getenv("NLP_DEVICE", "auto")

        # Które źródła aktywować
        self.enable_reddit     = bool(self.reddit_client_id and self.reddit_client_secret)
        self.enable_stocktwits = True     # zawsze (bez klucza)
        self.enable_news       = True     # zawsze (darmowy tier)
        self.enable_fear_greed = True     # zawsze (bezpłatny)
        self.enable_onchain    = bool(self.glassnode_api_key)

        # Polling
        self.poll_interval_sec = int(os.getenv("SENTIMENT_POLL_INTERVAL", "60"))


class SentimentAnalyzer:
    """
    Algorytm #1 — analizator sentymentu rynkowego.

    Pipeline:
        Kolektory (5 źródeł)
            → NLPProcessor (FinBERT / VADER)
            → SignalNormalizer (kompozyt ważony)
            → SentimentSignal (output)

    Interfejs dla Algo #2 i #3:
        signal = analyzer.get_latest_signal()
    """

    def __init__(self, config: Optional[SentimentAnalyzerConfig] = None):
        self.config = config or SentimentAnalyzerConfig()
        self._latest_signal: Optional[SentimentSignal] = None
        self._last_update: float = 0.0
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="sentiment")

        # --- NLP processor (współdzielony) ---
        self._nlp = NLPProcessor(
            use_finbert = self.config.use_finbert,
            device      = self.config.nlp_device,
        )

        # --- Kolektory ---
        self._collectors = self._init_collectors()

        # --- Normalizer ---
        self._normalizer = SignalNormalizer(symbol=self.config.symbol)

        logger.info(
            f"SentimentAnalyzer: zainicjalizowany | "
            f"symbol={self.config.symbol} | "
            f"aktywne źródła: {list(self._collectors.keys())}"
        )

    # ------------------------------------------------------------------
    # Inicjalizacja
    # ------------------------------------------------------------------

    def _init_collectors(self) -> dict:
        """Inicjalizuje aktywne kolektory."""
        collectors = {}
        cfg = self.config

        if cfg.enable_stocktwits:
            c = StockTwitsCollector()
            c.set_nlp_processor(self._nlp)
            collectors["stocktwits"] = c
            logger.info("Kolektor: StockTwits ✓")

        if cfg.enable_news:
            c = NewsCollector(api_key=cfg.cryptopanic_api_key)
            c.set_nlp_processor(self._nlp)
            collectors["news"] = c
            logger.info("Kolektor: CryptoPanic News ✓")

        if cfg.enable_fear_greed:
            collectors["fear_greed"] = FearGreedCollector()
            logger.info("Kolektor: Fear & Greed ✓")

        if cfg.enable_onchain:
            try:
                collectors["onchain"] = OnchainCollector(
                    api_key=cfg.glassnode_api_key
                )
                logger.info("Kolektor: Glassnode On-chain ✓")
            except ValueError as e:
                logger.warning(f"Kolektor: Glassnode pominięty — {e}")

        if cfg.enable_reddit:
            try:
                c = RedditCollector(
                    client_id     = cfg.reddit_client_id,
                    client_secret = cfg.reddit_client_secret,
                )
                c.set_nlp_processor(self._nlp)
                collectors["reddit"] = c
                logger.info("Kolektor: Reddit ✓")
            except (ValueError, ImportError) as e:
                logger.warning(f"Kolektor: Reddit pominięty — {e}")

        if not collectors:
            raise RuntimeError("Brak aktywnych kolektorów — sprawdź konfigurację API keys")

        return collectors

    # ------------------------------------------------------------------
    # Publiczny interfejs (dla Algo #2 i #3)
    # ------------------------------------------------------------------

    def get_latest_signal(self) -> Optional[SentimentSignal]:
        """
        Zwraca ostatni obliczony sygnał.
        Główny interfejs dla Algorytmu #2 i #3 — synchroniczny, nie blokuje.
        """
        return self._latest_signal

    def get_signal_sync(self) -> Optional[SentimentSignal]:
        """
        Synchroniczne pobranie i obliczenie nowego sygnału.
        Używaj gdy nie masz event loop (np. testy, skrypty).
        """
        return self._run_pipeline()

    async def get_signal_async(self) -> Optional[SentimentSignal]:
        """
        Asynchroniczne pobranie sygnału (nie blokuje event loop).
        Używaj w środowisku async (np. główna pętla tradingowa).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._run_pipeline)

    def update_weights(self, weights: dict) -> None:
        """
        Aktualizuje wagi źródeł — wywoływane przez Algorytm #4 (meta-RL).
        """
        self._normalizer.update_weights(weights)

    # ------------------------------------------------------------------
    # Pętla ciągłego monitorowania
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """
        Uruchamia ciągłe pobieranie sygnałów w tle.
        Wywołaj jako: asyncio.create_task(analyzer.start_polling())
        """
        self._running = True
        logger.info(f"SentimentAnalyzer: start polling (interval={self.config.poll_interval_sec}s)")

        # Warmup NLP model przed pierwszą iteracją
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._nlp.warmup)

        while self._running:
            try:
                signal = await self.get_signal_async()
                if signal:
                    self._latest_signal = signal
                    self._last_update   = time.time()
                    self._log_signal(signal)

            except Exception as e:
                logger.error(f"SentimentAnalyzer: błąd w pętli — {e}", exc_info=True)

            await asyncio.sleep(self.config.poll_interval_sec)

    def stop(self) -> None:
        """Zatrzymuje pętlę pollingu."""
        self._running = False
        self._executor.shutdown(wait=False)
        logger.info("SentimentAnalyzer: zatrzymany")

    # ------------------------------------------------------------------
    # Wewnętrzny pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> Optional[SentimentSignal]:
        """
        Wykonuje pełny pipeline:
        1. Pobierz sygnały ze wszystkich kolektorów (równolegle)
        2. Złóż w SentimentSignal przez normalizer
        """
        symbol = self.config.btc_symbol
        raw_signals = {}

        # Pobierz sygnały równolegle przez ThreadPoolExecutor
        futures = {}
        for name, collector in self._collectors.items():
            future = self._executor.submit(self._safe_collect, name, collector, symbol)
            futures[name] = future

        for name, future in futures.items():
            try:
                raw_signals[name] = future.result(timeout=15)
            except Exception as e:
                logger.warning(f"SentimentAnalyzer: timeout/błąd kolektor '{name}' — {e}")
                raw_signals[name] = None

        # Złóż sygnał
        signal = self._normalizer.compose(
            reddit_signal     = raw_signals.get("reddit"),
            stocktwits_signal = raw_signals.get("stocktwits"),
            news_signal       = raw_signals.get("news"),
            fear_greed_signal = raw_signals.get("fear_greed"),
            onchain_signal    = raw_signals.get("onchain"),
        )

        return signal

    def _safe_collect(self, name: str, collector, symbol: str):
        """Wywołuje get_signal() z obsługą błędów."""
        try:
            return collector.get_signal(symbol)
        except Exception as e:
            logger.error(f"SentimentAnalyzer: błąd kolektor '{name}' — {e}")
            return None

    def _log_signal(self, signal: SentimentSignal) -> None:
        """Loguje sygnał w czytelnym formacie."""
        direction_emoji = {
            "strongly_bullish": "🟢🟢",
            "bullish":          "🟢",
            "neutral":          "⚪",
            "bearish":          "🔴",
            "strongly_bearish": "🔴🔴",
        }.get(signal.direction.value, "❓")

        logger.info(
            f"[SENTIMENT] {direction_emoji} "
            f"score={signal.composite_score:+.3f} "
            f"conf={signal.confidence:.2f} "
            f"quality={signal.quality.value} "
            f"vel={signal.sentiment_velocity:+.3f} "
            f"session={signal.market_session} "
            f"dominant={signal.dominant_source} "
            f"tradeable={signal.is_tradeable}"
        )

        if signal.anomaly_detected:
            logger.warning(f"[SENTIMENT] ⚠️  Anomalia wykryta: {signal.anomaly_source}")

        if not signal.is_tradeable:
            logger.info(f"[SENTIMENT] ⛔ Sygnał nie-tradeable: {signal.skip_reason}")


# ---------------------------------------------------------------------------
# Szybki test manualny
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        print("\n" + "="*60)
        print("  SentimentAnalyzer — test manualny")
        print("="*60 + "\n")

        analyzer = SentimentAnalyzer()

        print("Pobieranie sygnału...\n")
        signal = await analyzer.get_signal_async()

        if signal:
            print(f"✅ Sygnał wygenerowany:")
            print(f"   ID:              {signal.signal_id}")
            print(f"   Composite score: {signal.composite_score:+.4f}")
            print(f"   Direction:       {signal.direction.value}")
            print(f"   Confidence:      {signal.confidence:.4f}")
            print(f"   Quality:         {signal.quality.value}")
            print(f"   Velocity:        {signal.sentiment_velocity:+.4f}")
            print(f"   Anomaly:         {signal.anomaly_detected} ({signal.anomaly_source})")
            print(f"   TF relevance:    1m={signal.tf_1m_relevance:.2f}  5m={signal.tf_5m_relevance:.2f}  15m={signal.tf_15m_relevance:.2f}")
            print(f"   Session:         {signal.market_session}")
            print(f"   Dominant source: {signal.dominant_source}")
            print(f"   Tradeable:       {signal.is_tradeable}")
            if signal.skip_reason:
                print(f"   Skip reason:     {signal.skip_reason}")

            print(f"\n   Składowe:")
            for src_name, src_sig in [
                ("stocktwits",  signal.stocktwits_signal),
                ("reddit",      signal.reddit_signal),
                ("news",        signal.news_signal),
                ("fear_greed",  signal.fear_greed_signal),
                ("onchain",     signal.onchain_signal),
            ]:
                if src_sig:
                    stale = " [STALE]" if src_sig.is_stale else ""
                    print(f"     {src_name:<15} score={src_sig.score:+.4f}  "
                          f"conf={src_sig.confidence:.4f}  "
                          f"n={src_sig.sample_count}{stale}")
                else:
                    print(f"     {src_name:<15} niedostępny")
        else:
            print("❌ Brak sygnału")

        analyzer.stop()

    asyncio.run(main())