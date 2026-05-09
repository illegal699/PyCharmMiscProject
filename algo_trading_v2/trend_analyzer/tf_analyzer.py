"""
trend_analyzer/tf_analyzer.py
-------------------------------
Analizuje pojedynczy timeframe i zwraca TimeframeAnalysis.
Łączy EMA, RSI, MACD, strukturę rynku i wolumen w jeden score.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from trend_analyzer.data_fetcher import BinanceDataFetcher, Candle
from trend_analyzer.indicators   import ema_latest, rsi, macd, market_structure, volume_analysis
from trend_analyzer.output_schema import (
    TimeframeAnalysis, TrendDirection, MarketPhase,
    score_to_direction,
)

logger = logging.getLogger(__name__)

# Wagi składowych w finalnym score
WEIGHTS = {
    "ema":       0.30,
    "rsi":       0.20,
    "macd":      0.25,
    "structure": 0.15,
    "volume":    0.10,
}


class TimeframeAnalyzer:
    """Analizuje jeden timeframe — zwraca TimeframeAnalysis."""

    def __init__(self, fetcher: BinanceDataFetcher):
        self._fetcher = fetcher

    def analyze(self, symbol: str, timeframe: str) -> Optional[TimeframeAnalysis]:
        candles = self._fetcher.get_candles(symbol, timeframe)
        if not candles or len(candles) < 50:
            logger.warning(f"TFAnalyzer: za mało świec dla {symbol} {timeframe}")
            return None

        closes  = [c.close for c in candles]
        price   = closes[-1]

        # --- EMA ---
        ema9  = ema_latest(closes, 9)
        ema21 = ema_latest(closes, 21)
        ema50 = ema_latest(closes, 50)

        if not all([ema9, ema21, ema50]):
            return None

        ema_score = self._ema_score(price, ema9, ema21, ema50)

        # --- RSI ---
        rsi_val   = rsi(closes, 14)
        rsi_score = self._rsi_score(rsi_val) if rsi_val else 0.0

        # --- MACD ---
        macd_result   = macd(closes)
        macd_val      = macd_result[0] if macd_result else 0.0
        macd_sig      = macd_result[1] if macd_result else 0.0
        macd_hist_val = macd_result[2] if macd_result else 0.0
        macd_score    = self._macd_score(macd_hist_val, macd_val, macd_sig)

        # --- Struktura rynku ---
        struct       = market_structure(candles)
        struct_score = self._structure_score(struct)

        # --- Wolumen ---
        vol         = volume_analysis(candles)
        vol_score   = self._volume_score(vol, ema_score)

        # --- Kompozyt ---
        composite = (
            ema_score    * WEIGHTS["ema"]       +
            rsi_score    * WEIGHTS["rsi"]       +
            macd_score   * WEIGHTS["macd"]      +
            struct_score * WEIGHTS["structure"] +
            vol_score    * WEIGHTS["volume"]
        )
        composite = max(-1.0, min(1.0, composite))
        strength  = abs(composite)

        # --- Faza rynku ---
        phase = self._market_phase(composite, struct, vol)

        return TimeframeAnalysis(
            timeframe    = timeframe,
            direction    = score_to_direction(composite),
            strength     = strength,
            score        = round(composite, 4),
            ema_fast     = round(ema9,  2),
            ema_slow     = round(ema21, 2),
            ema_trend    = round(ema50, 2),
            rsi          = round(rsi_val, 2) if rsi_val else 50.0,
            macd         = round(macd_val, 6),
            macd_signal  = round(macd_sig, 6),
            macd_hist    = round(macd_hist_val, 6),
            higher_highs = struct["higher_highs"],
            lower_lows   = struct["lower_lows"],
            last_high    = round(struct["last_high"], 2),
            last_low     = round(struct["last_low"],  2),
            volume_trend = vol["trend"],
            volume_ratio = vol["ratio"],
            phase        = phase,
            price        = round(price, 2),
            timestamp    = datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Scoring składowych
    # ------------------------------------------------------------------

    def _ema_score(self, price: float, ema9: float, ema21: float, ema50: float) -> float:
        """
        EMA score: -1.0 .. +1.0
        Idealne ustawienie bullish: cena > EMA9 > EMA21 > EMA50
        Idealne ustawienie bearish: cena < EMA9 < EMA21 < EMA50
        """
        score = 0.0

        # Cena vs EMA50 (trend główny)
        if price > ema50:
            score += 0.4
        else:
            score -= 0.4

        # EMA9 vs EMA21 (momentum)
        if ema9 > ema21:
            score += 0.3
        else:
            score -= 0.3

        # EMA21 vs EMA50 (trend średnioterminowy)
        if ema21 > ema50:
            score += 0.2
        else:
            score -= 0.2

        # Cena vs EMA9 (krótkoterminowy)
        if price > ema9:
            score += 0.1
        else:
            score -= 0.1

        return round(score, 4)

    def _rsi_score(self, rsi_val: float) -> float:
        """
        RSI score: -1.0 .. +1.0
        Oversold (<30) = bullish sygnał
        Overbought (>70) = bearish sygnał
        Momentum: >50 bullish, <50 bearish
        """
        if rsi_val >= 70:
            # Overbought — kontrariańsko bearish
            return -((rsi_val - 70) / 30) * 0.8
        elif rsi_val <= 30:
            # Oversold — kontrariańsko bullish
            return ((30 - rsi_val) / 30) * 0.8
        else:
            # Momentum: normalizuj 30-70 → -0.5 .. +0.5
            return round((rsi_val - 50) / 40, 4)

    def _macd_score(self, hist: float, macd_val: float, signal: float) -> float:
        """
        MACD score: -1.0 .. +1.0
        Histogram > 0 i rośnie = bullish
        Histogram < 0 i maleje = bearish
        Crossover = silny sygnał
        """
        score = 0.0

        # Histogram kierunek (podstawa)
        if hist > 0:
            score += 0.5
        elif hist < 0:
            score -= 0.5

        # MACD vs Signal (crossover)
        if macd_val > signal:
            score += 0.3
        else:
            score -= 0.3

        # MACD po stronie zera
        if macd_val > 0:
            score += 0.2
        else:
            score -= 0.2

        return round(max(-1.0, min(1.0, score)), 4)

    def _structure_score(self, struct: dict) -> float:
        """Struktura rynku: HH+HL = +1, LH+LL = -1."""
        if struct["bullish_structure"]:
            return 0.8
        elif struct["bearish_structure"]:
            return -0.8
        elif struct["higher_highs"]:
            return 0.4
        elif struct["lower_lows"]:
            return -0.4
        return 0.0

    def _volume_score(self, vol: dict, trend_score: float) -> float:
        """
        Wolumen potwierdza trend gdy rośnie w kierunku trendu.
        Wysoki wolumen contra-trend = ostrzeżenie.
        """
        ratio = vol["ratio"]
        trend = vol["trend"]

        if trend == "increasing" and ratio > 1.2:
            # Rosnący wolumen potwierdza kierunek trendu
            return 0.5 * (1 if trend_score >= 0 else -1)
        elif trend == "decreasing":
            # Spadający wolumen = słabnący trend
            return 0.0
        return 0.2 * (1 if trend_score >= 0 else -1)

    def _market_phase(self, score: float, struct: dict, vol: dict) -> MarketPhase:
        """Określa fazę rynku."""
        strength = abs(score)

        if strength < 0.25:
            return MarketPhase.CONSOLIDATION

        # Breakout: duży wolumen + wyjście z konsolidacji
        if vol["ratio"] > 1.5 and strength > 0.4:
            return MarketPhase.BREAKOUT

        # Reversal: struktura zmienia się
        if struct["higher_highs"] and score < -0.3:
            return MarketPhase.REVERSAL
        if struct["lower_lows"] and score > 0.3:
            return MarketPhase.REVERSAL

        return MarketPhase.TRENDING
