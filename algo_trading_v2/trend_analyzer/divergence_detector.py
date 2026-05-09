"""
trend_analyzer/divergence_detector.py
----------------------------------------
Wykrywanie regularnych dywergencji na RSI i MACD.

Regularna dywergencja Bullish:
    Cena robi Lower Low, wskaźnik robi Higher Low → sygnał odwrócenia w górę

Regularna dywergencja Bearish:
    Cena robi Higher High, wskaźnik robi Lower High → sygnał odwrócenia w dół

Algorytm:
1. Znajdź swing points (lokalne ekstrema) na cenie
2. Znajdź odpowiadające punkty na wskaźniku
3. Porównaj kierunki — jeśli sprzeczne = dywergencja
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from trend_analyzer.data_fetcher import BinanceDataFetcher, Candle
from trend_analyzer.indicators   import ema, rsi as calc_rsi, macd as calc_macd

logger = logging.getLogger(__name__)

# Ile świec tworzy swing point (im większy, tym rzadziej ale pewniej)
SWING_LOOKBACK = 5

# Minimalna odległość między swing points (żeby nie wykrywać szumu)
MIN_SWING_DISTANCE = 5

# Maksymalna odległość między swing points do porównania
MAX_SWING_DISTANCE = 50

# Minimalna różnica procentowa między punktami żeby uznać za dywergencję
MIN_DIVERGENCE_PCT = 0.003   # 0.3%


@dataclass
class SwingPoint:
    """Lokalny ekstremalny punkt."""
    index:     int
    price:     float
    indicator: float    # wartość wskaźnika w tym punkcie
    kind:      str      # "high" | "low"
    timestamp: Optional[datetime] = None


@dataclass
class Divergence:
    """Wykryta dywergencja."""
    type:        str          # "bullish" | "bearish"
    indicator:   str          # "rsi" | "macd"
    timeframe:   str          # "5m" | "15m"
    strength:    str          # "strong" | "medium" | "weak"
    confidence:  float        # 0.0 .. 1.0

    # Punkty dywergencji
    point_a:     SwingPoint   # starszy punkt
    point_b:     SwingPoint   # nowszy punkt

    # Opis
    price_desc:  str          # np. "LL: 94500 → 94200"
    indic_desc:  str          # np. "HL: 28.5 → 31.2"
    description: str          # pełny opis

    # Sugestia dla Algo #3
    suggested_side: str       # "long" | "short"
    score:          float     # -1.0 .. +1.0 (ujemny = bearish, dodatni = bullish)


@dataclass
class DivergenceResult:
    """Wynik analizy dywergencji dla jednego TF."""
    timeframe:    str
    timestamp:    datetime
    divergences:  list = field(default_factory=list)   # lista Divergence
    has_bullish:  bool = False
    has_bearish:  bool = False
    strongest:    Optional[object] = None   # najsilniejsza dywergencja
    score:        float = 0.0               # -1.0 .. +1.0


class DivergenceDetector:
    """
    Wykrywa regularne dywergencje RSI i MACD na timeframe'ach 5m i 15m.
    """

    def __init__(self, fetcher: BinanceDataFetcher):
        self._fetcher = fetcher

    # ------------------------------------------------------------------
    # Publiczny interfejs
    # ------------------------------------------------------------------

    def detect(self, symbol: str) -> dict[str, DivergenceResult]:
        """
        Wykrywa dywergencje na 5m i 15m.
        Zwraca słownik: {"5m": DivergenceResult, "15m": DivergenceResult}
        """
        results = {}

        for tf in ["5m", "15m"]:
            candles = self._fetcher.get_candles(symbol, tf, limit=100)
            if not candles or len(candles) < 50:
                logger.warning(f"Divergence: za mało świec dla {symbol} {tf}")
                continue

            result = self._analyze_tf(candles, tf)
            results[tf] = result
            self._log_result(result)

        return results

    # ------------------------------------------------------------------
    # Analiza TF
    # ------------------------------------------------------------------

    def _analyze_tf(self, candles: list[Candle], tf: str) -> DivergenceResult:
        closes = [c.close for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]

        # Oblicz wskaźniki
        rsi_values  = self._compute_rsi_series(closes)
        macd_values = self._compute_macd_series(closes)

        divergences = []

        # RSI dywergencje
        if rsi_values:
            divs = self._find_divergences(
                closes, highs, lows, rsi_values, "rsi", tf
            )
            divergences.extend(divs)

        # MACD (histogram) dywergencje
        if macd_values:
            divs = self._find_divergences(
                closes, highs, lows, macd_values, "macd", tf
            )
            divergences.extend(divs)

        # Sortuj po sile (confidence malejąco)
        divergences.sort(key=lambda d: d.confidence, reverse=True)

        has_bullish = any(d.type == "bullish" for d in divergences)
        has_bearish = any(d.type == "bearish" for d in divergences)
        strongest   = divergences[0] if divergences else None

        # Score: bullish dywergencje dają +, bearish dają -
        score = 0.0
        for d in divergences:
            weight = d.confidence * (1.0 if d.strength == "strong" else
                                     0.6 if d.strength == "medium" else 0.3)
            score += weight if d.type == "bullish" else -weight
        score = max(-1.0, min(1.0, score))

        return DivergenceResult(
            timeframe   = tf,
            timestamp   = datetime.now(tz=timezone.utc),
            divergences = divergences,
            has_bullish = has_bullish,
            has_bearish = has_bearish,
            strongest   = strongest,
            score       = round(score, 4),
        )

    # ------------------------------------------------------------------
    # Znajdowanie dywergencji
    # ------------------------------------------------------------------

    def _find_divergences(
        self,
        closes:    list[float],
        highs:     list[float],
        lows:      list[float],
        indicator: list[float],
        indic_name: str,
        tf:        str,
    ) -> list[Divergence]:
        """
        Główna logika wykrywania dywergencji.
        Dla bullish: szuka par Lower Low na cenie + Higher Low na wskaźniku
        Dla bearish: szuka par Higher High na cenie + Lower High na wskaźniku
        """
        divergences = []

        # Wyrównaj długości (wskaźnik może być krótszy)
        offset = len(closes) - len(indicator)
        aligned_closes = closes[offset:]
        aligned_highs  = highs[offset:]
        aligned_lows   = lows[offset:]
        n = len(indicator)

        # Znajdź swing highs i swing lows
        swing_highs = self._find_swings(aligned_highs, indicator, "high")
        swing_lows  = self._find_swings(aligned_lows,  indicator, "low")

        # Bullish: Lower Low na cenie + Higher Low na wskaźniku
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                a = swing_lows[i]
                b = swing_lows[j]

                if b.index - a.index < MIN_SWING_DISTANCE:
                    continue
                if b.index - a.index > MAX_SWING_DISTANCE:
                    continue

                # Cena robi LL (b.price < a.price)
                price_diff = (a.price - b.price) / a.price
                if price_diff < MIN_DIVERGENCE_PCT:
                    continue

                # Wskaźnik robi HL (b.indicator > a.indicator)
                indic_diff = (b.indicator - a.indicator) / max(abs(a.indicator), 0.001)
                if indic_diff < MIN_DIVERGENCE_PCT:
                    continue

                # Dywergencja bullish!
                div = self._build_divergence(
                    "bullish", indic_name, tf, a, b,
                    price_diff, indic_diff
                )
                divergences.append(div)

        # Bearish: Higher High na cenie + Lower High na wskaźniku
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                a = swing_highs[i]
                b = swing_highs[j]

                if b.index - a.index < MIN_SWING_DISTANCE:
                    continue
                if b.index - a.index > MAX_SWING_DISTANCE:
                    continue

                # Cena robi HH (b.price > a.price)
                price_diff = (b.price - a.price) / a.price
                if price_diff < MIN_DIVERGENCE_PCT:
                    continue

                # Wskaźnik robi LH (b.indicator < a.indicator)
                indic_diff = (a.indicator - b.indicator) / max(abs(a.indicator), 0.001)
                if indic_diff < MIN_DIVERGENCE_PCT:
                    continue

                # Dywergencja bearish!
                div = self._build_divergence(
                    "bearish", indic_name, tf, a, b,
                    price_diff, indic_diff
                )
                divergences.append(div)

        # Zostaw tylko najnowsze (ostatnie 3 na wskaźnik)
        divergences = sorted(divergences, key=lambda d: d.point_b.index, reverse=True)
        return divergences[:3]

    def _build_divergence(
        self,
        div_type:   str,
        indic_name: str,
        tf:         str,
        a:          SwingPoint,
        b:          SwingPoint,
        price_diff: float,
        indic_diff: float,
    ) -> Divergence:
        """Buduje obiekt Divergence z wykrytych punktów."""

        # Siła dywergencji
        combined_diff = (price_diff + indic_diff) / 2
        if combined_diff > 0.03:
            strength    = "strong"
            base_conf   = 0.80
        elif combined_diff > 0.01:
            strength    = "medium"
            base_conf   = 0.60
        else:
            strength    = "weak"
            base_conf   = 0.40

        # Bonus za bliskość (świeższa dywergencja = bardziej wiarygodna)
        recency_bonus = max(0, (MAX_SWING_DISTANCE - (b.index - a.index))
                           / MAX_SWING_DISTANCE) * 0.15
        confidence    = min(base_conf + recency_bonus, 0.95)

        # Opisy
        if div_type == "bullish":
            price_desc = f"LL: {a.price:,.2f} → {b.price:,.2f}"
            indic_desc = f"HL: {a.indicator:.2f} → {b.indicator:.2f}"
            side       = "long"
            score      = confidence
        else:
            price_desc = f"HH: {a.price:,.2f} → {b.price:,.2f}"
            indic_desc = f"LH: {a.indicator:.2f} → {b.indicator:.2f}"
            side       = "short"
            score      = -confidence

        description = (
            f"{div_type.upper()} divergence ({indic_name.upper()}) na {tf} | "
            f"{strength} | {price_desc} | {indic_desc}"
        )

        return Divergence(
            type           = div_type,
            indicator      = indic_name,
            timeframe      = tf,
            strength       = strength,
            confidence     = round(confidence, 4),
            point_a        = a,
            point_b        = b,
            price_desc     = price_desc,
            indic_desc     = indic_desc,
            description    = description,
            suggested_side = side,
            score          = round(score, 4),
        )

    # ------------------------------------------------------------------
    # Swing points
    # ------------------------------------------------------------------

    def _find_swings(
        self,
        prices:    list[float],
        indicator: list[float],
        kind:      str,
    ) -> list[SwingPoint]:
        """
        Znajduje lokalne ekstrema (swing highs lub swing lows).
        kind: "high" | "low"
        """
        swings = []
        n      = len(prices)
        lb     = SWING_LOOKBACK

        for i in range(lb, n - lb):
            window_prices = prices[i - lb: i + lb + 1]
            center_price  = prices[i]

            if kind == "high":
                if center_price != max(window_prices):
                    continue
            else:
                if center_price != min(window_prices):
                    continue

            # Bezpieczny dostęp do wskaźnika
            if i >= len(indicator):
                continue
            indic_val = indicator[i]

            swings.append(SwingPoint(
                index     = i,
                price     = center_price,
                indicator = indic_val,
                kind      = kind,
            ))

        return swings

    # ------------------------------------------------------------------
    # Obliczanie wskaźników jako seria
    # ------------------------------------------------------------------

    def _compute_rsi_series(self, closes: list[float], period: int = 14) -> list[float]:
        """Oblicza RSI dla każdego punktu (seria wartości)."""
        if len(closes) < period + 1:
            return []

        gains  = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
        losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_series = []

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs  = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi_val = 100 - (100 / (1 + rs))
            rsi_series.append(round(rsi_val, 4))

        return rsi_series

    def _compute_macd_series(
        self,
        closes: list[float],
        fast: int = 12, slow: int = 26, signal: int = 9
    ) -> list[float]:
        """Oblicza serię histogramów MACD."""
        if len(closes) < slow + signal:
            return []

        k_fast = 2 / (fast + 1)
        k_slow = 2 / (slow + 1)
        k_sig  = 2 / (signal + 1)

        ema_fast = sum(closes[:fast]) / fast
        ema_slow = sum(closes[:slow]) / slow

        macd_line = []
        for i in range(slow, len(closes)):
            ema_fast = closes[i] * k_fast + ema_fast * (1 - k_fast)
            ema_slow = closes[i] * k_slow + ema_slow * (1 - k_slow)
            macd_line.append(ema_fast - ema_slow)

        if len(macd_line) < signal:
            return []

        sig_line = sum(macd_line[:signal]) / signal
        hist     = []
        for i in range(signal, len(macd_line)):
            sig_line = macd_line[i] * k_sig + sig_line * (1 - k_sig)
            hist.append(round(macd_line[i] - sig_line, 8))

        return hist

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_result(self, result: DivergenceResult) -> None:
        if not result.divergences:
            logger.debug(f"Divergence {result.timeframe}: brak dywergencji")
            return

        for d in result.divergences:
            emoji = "🟢" if d.type == "bullish" else "🔴"
            logger.info(
                f"[DIVERGENCE] {emoji} {d.description} "
                f"conf={d.confidence:.2f}"
            )


# ---------------------------------------------------------------------------
# Test manualny
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    fetcher  = BinanceDataFetcher()
    detector = DivergenceDetector(fetcher)
    results  = detector.detect("BTCUSDT")

    for tf, result in results.items():
        print(f"\n── {tf} ──────────────────────────────────")
        print(f"   Score:      {result.score:+.4f}")
        print(f"   Bullish:    {result.has_bullish}")
        print(f"   Bearish:    {result.has_bearish}")
        if result.divergences:
            print(f"   Dywergencje ({len(result.divergences)}):")
            for d in result.divergences:
                emoji = "🟢" if d.type == "bullish" else "🔴"
                print(f"     {emoji} [{d.strength}] {d.indicator.upper()} "
                      f"conf={d.confidence:.2f}")
                print(f"        Cena:  {d.price_desc}")
                print(f"        Wsk.:  {d.indic_desc}")
        else:
            print("   Brak dywergencji")
