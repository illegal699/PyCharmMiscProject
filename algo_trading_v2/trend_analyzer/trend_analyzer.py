"""
trend_analyzer/trend_analyzer.py
----------------------------------
Algorytm #2 — TrendAnalyzer.
Hierarchiczna analiza trendu: 15m → 5m → 1m.

15m: nadrzędny kierunek rynku
5m:  kierunek wejścia w pozycję
1m:  precyzja wejścia

Interfejs dla Algo #3:
    signal = analyzer.get_latest_signal()
    signal = await analyzer.get_signal_async()
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from trend_analyzer.data_fetcher       import BinanceDataFetcher
from trend_analyzer.divergence_detector import DivergenceDetector
from trend_analyzer.output_schema import (
    TrendSignal, TrendDirection, MarketPhase,
    classify_trend_strength, score_to_direction,
)
from trend_analyzer.tf_analyzer   import TimeframeAnalyzer

logger = logging.getLogger(__name__)

HISTORY_SIZE = 20


class TrendAnalyzer:
    """
    Algorytm #2 — hierarchiczny analizator trendu.

    Interfejs dla Algo #3:
        signal = analyzer.get_latest_signal()     # nie blokuje
        signal = analyzer.get_signal_sync()       # blokuje
        signal = await analyzer.get_signal_async()
    """

    def __init__(self, symbol: str = "BTCUSDT", poll_interval: int = 30):
        self.symbol        = symbol
        self.poll_interval = poll_interval

        self._fetcher      = BinanceDataFetcher()
        self._tf_analyzer  = TimeframeAnalyzer(self._fetcher)
        self._divergence   = DivergenceDetector(self._fetcher)
        self._executor     = ThreadPoolExecutor(max_workers=3, thread_name_prefix="trend")

        self._latest_signal: Optional[TrendSignal] = None
        self._history = deque(maxlen=HISTORY_SIZE)
        self._running = False

        logger.info(f"TrendAnalyzer: gotowy | symbol={symbol}")

    # ------------------------------------------------------------------
    # Publiczny interfejs
    # ------------------------------------------------------------------

    def get_latest_signal(self) -> Optional[TrendSignal]:
        """Zwraca ostatni sygnał — nie blokuje."""
        return self._latest_signal

    def get_signal_sync(self) -> Optional[TrendSignal]:
        """Pobiera nowy sygnał synchronicznie."""
        return self._pipeline()

    async def get_signal_async(self) -> Optional[TrendSignal]:
        """Pobiera nowy sygnał asynchronicznie."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._pipeline)

    async def start_polling(self) -> None:
        """Ciągłe pobieranie sygnałów w tle."""
        self._running = True
        logger.info(f"TrendAnalyzer: start polling co {self.poll_interval}s")

        while self._running:
            try:
                signal = await self.get_signal_async()
                if signal:
                    self._latest_signal = signal
                    self._log(signal)
            except Exception as e:
                logger.error(f"TrendAnalyzer: błąd pętli — {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        logger.info("TrendAnalyzer: zatrzymany")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _pipeline(self) -> Optional[TrendSignal]:
        """
        Hierarchiczna analiza:
        1. Pobierz analizę 15m, 5m, 1m równolegle
        2. Złóż w TrendSignal z hierarchią
        """
        # Pobierz wszystkie TF równolegle + dywergencje
        f15  = self._executor.submit(self._tf_analyzer.analyze, self.symbol, "15m")
        f5   = self._executor.submit(self._tf_analyzer.analyze, self.symbol, "5m")
        f1   = self._executor.submit(self._tf_analyzer.analyze, self.symbol, "1m")
        fdiv = self._executor.submit(self._divergence.detect, self.symbol)

        tf15 = f15.result(timeout=15)
        tf5  = f5.result(timeout=15)
        tf1  = f1.result(timeout=15)
        divs = fdiv.result(timeout=20)

        if not tf15 or not tf5 or not tf1:
            logger.warning("TrendAnalyzer: brak danych dla jednego z TF")
            return None

        return self._compose(tf15, tf5, tf1, divs)

    def _compose(self, tf15, tf5, tf1, divs: dict = None) -> TrendSignal:
        """
        Składa analizy 3 TF + dywergencje w jeden TrendSignal.
        """
        # Ważony score trendu
        composite = (
            tf15.score * 0.50 +
            tf5.score  * 0.35 +
            tf1.score  * 0.15
        )
        composite = round(max(-1.0, min(1.0, composite)), 4)

        # Dywergencje
        div_5m  = divs.get("5m")  if divs else None
        div_15m = divs.get("15m") if divs else None

        # Score dywergencji (może modyfikować końcowy wynik)
        div_score = 0.0
        if div_5m:  div_score += div_5m.score  * 0.4
        if div_15m: div_score += div_15m.score * 0.6
        div_score = round(max(-1.0, min(1.0, div_score)), 4)

        # Dywergencja kontra trend = ostrzeżenie (nie zmieniamy trendu, ale obniżamy confidence)
        div_contra_trend = (
            (composite > 0.2 and div_score < -0.3) or
            (composite < -0.2 and div_score > 0.3)
        )

        # Kierunki
        primary_direction = score_to_direction(tf15.score)
        entry_direction   = score_to_direction(tf5.score)

        # Zgodność TF
        alignment, alignment_desc = self._tf_alignment(tf15, tf5, tf1)

        # Pewność
        base_conf   = abs(composite) * 0.7
        align_bonus = alignment * 0.3
        div_penalty = -0.15 if div_contra_trend else 0.0
        confidence  = round(min(base_conf + align_bonus + div_penalty, 0.95), 4)

        # Siła trendu
        strength = classify_trend_strength(abs(composite))

        # Faza
        phase = tf15.phase

        # Czy tradeable
        tradeable, skip, side = self._tradeable(
            composite, alignment, primary_direction, entry_direction,
            tf15, tf5, div_contra_trend
        )

        # Poziom unieważnienia
        invalidation = self._invalidation_level(side, tf15, tf5)

        # Historia
        self._history.append(composite)

        return TrendSignal(
            timestamp          = datetime.now(tz=timezone.utc),
            symbol             = self.symbol,
            signal_id          = str(uuid.uuid4())[:8],
            primary_direction  = primary_direction,
            entry_direction    = entry_direction,
            score              = composite,
            strength           = strength,
            confidence         = confidence,
            tf_15m             = tf15,
            tf_5m              = tf5,
            tf_1m              = tf1,
            tf_alignment       = round(alignment, 4),
            tf_alignment_desc  = alignment_desc,
            divergences_5m     = div_5m,
            divergences_15m    = div_15m,
            divergence_score   = div_score,
            is_tradeable       = tradeable,
            skip_reason        = skip,
            suggested_side     = side,
            invalidation_level = invalidation,
            market_phase       = phase,
            current_price      = tf1.price,
            score_history      = list(self._history)[-10:],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tf_alignment(self, tf15, tf5, tf1) -> tuple[float, str]:
        """
        Mierzy zgodność między timeframe'ami.
        1.0 = wszystkie w tym samym kierunku
        0.0 = sprzeczne
        """
        def sign(s): return 1 if s > 0.1 else (-1 if s < -0.1 else 0)

        s15, s5, s1 = sign(tf15.score), sign(tf5.score), sign(tf1.score)

        arrows = {1: "↑", -1: "↓", 0: "→"}
        desc   = f"15m{arrows[s15]} 5m{arrows[s5]} 1m{arrows[s1]}"

        if s15 == s5 == s1 and s15 != 0:
            return 1.0, desc   # pełna zgodność
        elif s15 == s5 and s15 != 0:
            return 0.75, desc  # 15m i 5m zgodne
        elif s15 != 0 and s15 == s1:
            return 0.5, desc
        elif s15 != 0 and s5 != 0 and s15 != s5:
            return 0.1, desc   # 15m i 5m sprzeczne — nie handluj
        else:
            return 0.3, desc

    def _tradeable(self, composite, alignment, primary, entry, tf15, tf5, div_contra_trend=False):
        """
        Decyduje czy warunki są sprzyjające do handlu.
        Zwraca (tradeable, skip_reason, side).
        """
        # Za słaby sygnał
        if abs(composite) < 0.15:
            return False, "weak_signal", None

        # 15m i 5m sprzeczne — zbyt ryzykowne
        if alignment < 0.2:
            return False, "tf_conflict", None

        # Konsolidacja na 15m — brak kierunku
        if tf15.phase == MarketPhase.CONSOLIDATION and abs(tf15.score) < 0.25:
            return False, "consolidation_15m", None

        # RSI ekstremalny contra trend
        if composite > 0.3 and tf15.rsi > 78:
            return False, "rsi_overbought_15m", None
        if composite < -0.3 and tf15.rsi < 22:
            return False, "rsi_oversold_15m", None

        # Dywergencja contra trend — obniż pewność ale nie blokuj
        if div_contra_trend:
            side = "long" if composite > 0.15 else "short" if composite < -0.15 else None
            return True, "divergence_warning", side

        # Określ stronę
        if composite > 0.15:
            side = "long"
        elif composite < -0.15:
            side = "short"
        else:
            side = None

        return True, None, side

    def _invalidation_level(self, side, tf15, tf5) -> Optional[float]:
        """
        Poziom cenowy który unieważnia sygnał.
        Long: poniżej ostatniego swing low z 5m
        Short: powyżej ostatniego swing high z 5m
        """
        if side == "long":
            return round(min(tf5.last_low, tf15.last_low), 2)
        elif side == "short":
            return round(max(tf5.last_high, tf15.last_high), 2)
        return None

    def _log(self, s: TrendSignal) -> None:
        emoji = {
            "strong_up": "🟢🟢", "up": "🟢",
            "balance":   "⚪",
            "down": "🔴", "strong_down": "🔴🔴",
        }
        e = emoji.get(s.primary_direction.value, "❓")
        logger.info(
            f"[TREND] {e} score={s.score:+.3f}  "
            f"conf={s.confidence:.2f}  "
            f"align={s.tf_alignment_desc}  "
            f"side={s.suggested_side}  "
            f"phase={s.market_phase.value}  "
            f"price={s.current_price}"
        )
        if not s.is_tradeable:
            logger.info(f"[TREND] ⛔ skip: {s.skip_reason}")


# ---------------------------------------------------------------------------
# Test manualny
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        print("\n" + "="*55)
        print("  TrendAnalyzer — test")
        print("="*55 + "\n")

        analyzer = TrendAnalyzer(symbol="BTCUSDT")
        print("Pobieranie danych z Binance...\n")

        signal = await analyzer.get_signal_async()

        if signal:
            print(f"✅ Sygnał trendu:")
            print(f"   Cena:            {signal.current_price}")
            print(f"   Kierunek (15m):  {signal.primary_direction.value}")
            print(f"   Kierunek (5m):   {signal.entry_direction.value}")
            print(f"   Score:           {signal.score:+.4f}")
            print(f"   Strength:        {signal.strength.value}")
            print(f"   Confidence:      {signal.confidence:.4f}")
            print(f"   Alignment:       {signal.tf_alignment_desc}")
            print(f"   Faza:            {signal.market_phase.value}")
            print(f"   Sugerowana str.: {signal.suggested_side}")
            print(f"   Invalidation:    {signal.invalidation_level}")
            print(f"   Tradeable:       {signal.is_tradeable}")
            if signal.skip_reason:
                print(f"   Skip reason:     {signal.skip_reason}")

            print(f"\n   Per timeframe:")
            for tf in [signal.tf_15m, signal.tf_5m, signal.tf_1m]:
                print(f"   {tf.timeframe:>3}  score={tf.score:+.3f}  "
                      f"rsi={tf.rsi:.1f}  "
                      f"ema9={tf.ema_fast:.1f}  ema21={tf.ema_slow:.1f}  "
                      f"vol={tf.volume_ratio:.2f}x  "
                      f"phase={tf.phase.value}")
        else:
            print("❌ Brak sygnału")

        analyzer.stop()

    asyncio.run(main())
