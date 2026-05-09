"""
trader/feature_builder.py
---------------------------
Buduje wektor cech (observation) dla agenta RL z danych rynkowych.
Wszystkie cechy są znormalizowane do zakresu [-1, +1] lub [0, 1].

Cechy są konfigurowane z dashboardu przez FeatureConfig.
"""

from dataclasses import dataclass, field
import math
from typing import Optional


@dataclass
class FeatureConfig:
    """
    Konfiguracja cech wejściowych — edytowalna z dashboardu (checklist).
    True = cecha włączona, False = wyłączona.
    """

    # ── Wskaźniki cenowe ─────────────────────────────────────────────
    use_ema_9:       bool = True   # EMA 9 znormalizowana do ceny
    use_ema_21:      bool = True   # EMA 21
    use_ema_50:      bool = True   # EMA 50
    use_ema_cross:   bool = True   # Krzyżowanie EMA (9 vs 21, 21 vs 50)

    use_rsi_1m:      bool = True   # RSI 14 na 1m
    use_rsi_5m:      bool = True   # RSI 14 na 5m
    use_rsi_15m:     bool = True   # RSI 14 na 15m

    use_macd_hist:   bool = True   # Histogram MACD (5m)
    use_macd_cross:  bool = True   # Czy MACD właśnie skrzyżował signal line

    # ── Struktura rynku ──────────────────────────────────────────────
    use_hh_hl:       bool = True   # Higher Highs / Higher Lows (bullish structure)
    use_ll_lh:       bool = True   # Lower Lows / Lower Highs (bearish structure)
    use_swing_dist:  bool = True   # Odległość ceny od ostatniego swing high/low

    # ── Wolumen ──────────────────────────────────────────────────────
    use_volume_ratio:   bool = True  # Wolumen / średni wolumen (20)
    use_volume_trend:   bool = True  # Trend wolumenu (rosnący/malejący)
    use_volume_delta:   bool = False # Delta wolumenu (buy vol - sell vol) — wymaga danych tick

    # ── Buy side / Sell side ─────────────────────────────────────────
    use_buy_volume:     bool = True  # Szacowany wolumen kupna (świece bullish)
    use_sell_volume:    bool = True  # Szacowany wolumen sprzedaży (świece bearish)
    use_cvd:            bool = True  # Cumulative Volume Delta (skumulowana różnica)
    use_large_candles:  bool = True  # Czy obecna świeca jest dużą świecą (>1.5x avg)

    # ── Sygnały z Algo #1 i #2 ───────────────────────────────────────
    use_sentiment_score:  bool = True  # Score sentymentu z Algo #1
    use_trend_score_5m:   bool = True  # Score trendu 5m z Algo #2
    use_trend_score_15m:  bool = True  # Score trendu 15m z Algo #2
    use_tf_alignment:     bool = True  # Zgodność timeframe'ów

    # ── Dywergencje ──────────────────────────────────────────────────
    use_divergence_score:    bool = True  # Score dywergencji
    use_divergence_bullish:  bool = True  # Czy jest bullish dywergencja (0/1)
    use_divergence_bearish:  bool = True  # Czy jest bearish dywergencja (0/1)
    use_divergence_strength: bool = True  # Siła dywergencji (0=brak, 0.5=medium, 1=strong)

    # ── Stan pozycji ─────────────────────────────────────────────────
    use_position_side:      bool = True  # Aktualna strona pozycji (-1=short, 0=brak, 1=long)
    use_position_pnl:       bool = True  # % P&L aktualnej pozycji
    use_position_duration:  bool = True  # Liczba świec w pozycji (znormalizowana)
    use_candles_since_close: bool = True # Ile świec od ostatniego zamknięcia

    # Cechy które zwracają 2 wartości zamiast 1
    DOUBLE_FEATURES = frozenset({
        "use_ema_cross",   # EMA9>EMA21 + EMA21>EMA50
        "use_hh_hl",       # higher_highs + higher_lows
        "use_ll_lh",       # lower_lows + lower_highs
        "use_swing_dist",  # dist_high + dist_low
    })

    def active_features(self) -> list[str]:
        """Zwraca listę nazw aktywnych cech."""
        return [k for k, v in self.__dict__.items()
                if k.startswith("use_") and v is True]

    def feature_count(self) -> int:
        """Zwraca dokładną liczbę wartości w wektorze cech."""
        count = 0
        for k in self.active_features():
            count += 2 if k in self.DOUBLE_FEATURES else 1
        return count

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k.startswith("use_")}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, bool(v))
        return obj


# Opisy cech do wyświetlenia w dashboardzie
FEATURE_DESCRIPTIONS = {
    "use_ema_9":              ("EMA 9",              "Szybka średnia krocząca — momentum krótkoterminowe"),
    "use_ema_21":             ("EMA 21",             "Średnia krocząca — trend krótkoterminowy"),
    "use_ema_50":             ("EMA 50",             "Wolna średnia krocząca — trend główny"),
    "use_ema_cross":          ("EMA Krzyżowanie",    "Sygnał gdy EMA9 przecina EMA21 lub EMA21 przecina EMA50"),
    "use_rsi_1m":             ("RSI 1m",             "Relative Strength Index na timeframe 1 minuta"),
    "use_rsi_5m":             ("RSI 5m",             "Relative Strength Index na timeframe 5 minut"),
    "use_rsi_15m":            ("RSI 15m",            "Relative Strength Index na timeframe 15 minut"),
    "use_macd_hist":          ("MACD Histogram",     "Siła i kierunek momentum — różnica MACD i signal line"),
    "use_macd_cross":         ("MACD Cross",         "Sygnał crossover linii MACD i signal"),
    "use_hh_hl":              ("Higher High/Low",    "Bullish struktura rynku: wyższe szczyty i wyższe dołki"),
    "use_ll_lh":              ("Lower Low/High",     "Bearish struktura rynku: niższe dołki i niższe szczyty"),
    "use_swing_dist":         ("Odległość Swing",    "Dystans ceny od ostatniego swing high i swing low"),
    "use_volume_ratio":       ("Wolumen Ratio",      "Aktualny wolumen vs średni wolumen z 20 świec"),
    "use_volume_trend":       ("Trend Wolumenu",     "Czy wolumen rośnie czy maleje w ostatnich świecach"),
    "use_volume_delta":       ("Volume Delta",       "Różnica buy/sell wolumenu (wymaga danych tick)"),
    "use_buy_volume":         ("Buy Volume",         "Szacowany wolumen kupna (świece bycze)"),
    "use_sell_volume":        ("Sell Volume",        "Szacowany wolumen sprzedaży (świece niedźwiedzie)"),
    "use_cvd":                ("CVD",                "Cumulative Volume Delta — narastający balans kupno/sprzedaż"),
    "use_large_candles":      ("Duże Świece",        "Czy świeca jest znacznie większa niż średnia"),
    "use_sentiment_score":    ("Sentyment Score",    "Wynik Algorytmu #1 — Fear&Greed + Google Trends"),
    "use_trend_score_5m":     ("Trend Score 5m",     "Wynik Algorytmu #2 na timeframe 5 minut"),
    "use_trend_score_15m":    ("Trend Score 15m",    "Wynik Algorytmu #2 na timeframe 15 minut"),
    "use_tf_alignment":       ("TF Alignment",       "Stopień zgodności sygnałów między timeframe'ami"),
    "use_divergence_score":   ("Dywergencja Score",  "Łączny score dywergencji RSI i MACD"),
    "use_divergence_bullish": ("Dywergencja Bullish","Czy wykryto bullish dywergencję (medium/strong)"),
    "use_divergence_bearish": ("Dywergencja Bearish","Czy wykryto bearish dywergencję (medium/strong)"),
    "use_divergence_strength":("Siła Dywergencji",   "Siła wykrytej dywergencji: 0=brak, 0.5=medium, 1=strong"),
    "use_position_side":      ("Strona Pozycji",     "Aktualna pozycja: -1=short, 0=brak, +1=long"),
    "use_position_pnl":       ("P&L Pozycji",        "Aktualny zysk/strata otwartej pozycji w %"),
    "use_position_duration":  ("Czas w Pozycji",     "Ile świec agent jest w aktualnej pozycji"),
    "use_candles_since_close":("Świece od Zamknięcia","Ile świec minęło od ostatniego zamknięcia pozycji"),
}

# Grupowanie cech dla UI
FEATURE_GROUPS = {
    "📈 Wskaźniki cenowe": [
        "use_ema_9", "use_ema_21", "use_ema_50", "use_ema_cross",
        "use_rsi_1m", "use_rsi_5m", "use_rsi_15m",
        "use_macd_hist", "use_macd_cross",
    ],
    "🏗️ Struktura rynku": [
        "use_hh_hl", "use_ll_lh", "use_swing_dist",
    ],
    "📊 Wolumen": [
        "use_volume_ratio", "use_volume_trend", "use_volume_delta",
        "use_buy_volume", "use_sell_volume", "use_cvd", "use_large_candles",
    ],
    "🤖 Sygnały Algo #1 i #2": [
        "use_sentiment_score", "use_trend_score_5m",
        "use_trend_score_15m", "use_tf_alignment",
    ],
    "🔀 Dywergencje": [
        "use_divergence_score", "use_divergence_bullish",
        "use_divergence_bearish", "use_divergence_strength",
    ],
    "💼 Stan pozycji": [
        "use_position_side", "use_position_pnl",
        "use_position_duration", "use_candles_since_close",
    ],
}


class FeatureBuilder:
    """
    Buduje wektor obserwacji dla agenta RL.
    Przyjmuje surowe dane i zwraca znormalizowany numpy array.
    """

    def __init__(self, config: FeatureConfig):
        self.cfg = config

    def build(self, market_state: dict) -> list[float]:
        """
        Buduje wektor cech z market_state.

        market_state zawiera wszystkie dostępne dane:
        {
            "close": float, "open": float, "high": float, "low": float,
            "volume": float,
            "ema_9": float, "ema_21": float, "ema_50": float,
            "rsi_1m": float, "rsi_5m": float, "rsi_15m": float,
            "macd_hist": float, "macd_line": float, "macd_signal": float,
            "higher_highs": bool, "lower_lows": bool,
            "last_high": float, "last_low": float,
            "volume_ratio": float, "volume_trend": str,
            "buy_volume": float, "sell_volume": float, "cvd": float,
            "sentiment_score": float,
            "trend_score_5m": float, "trend_score_15m": float,
            "tf_alignment": float,
            "divergence_score": float,
            "divergence_bullish": bool, "divergence_bearish": bool,
            "divergence_strength": str,
            "position_side": str,       # "long" | "short" | None
            "position_pnl": float,
            "candles_in_position": int,
            "candles_since_close": int,
            "avg_candle_size": float,
        }
        """
        features = []
        cfg = self.cfg
        ms  = market_state
        price = ms.get("close", 1.0)

        def safe(key, default=0.0):
            return ms.get(key, default)

        # ── Wskaźniki cenowe ─────────────────────────────────────────
        if cfg.use_ema_9:
            features.append(self._norm_price_ratio(safe("ema_9"), price))
        if cfg.use_ema_21:
            features.append(self._norm_price_ratio(safe("ema_21"), price))
        if cfg.use_ema_50:
            features.append(self._norm_price_ratio(safe("ema_50"), price))
        if cfg.use_ema_cross:
            e9, e21, e50 = safe("ema_9"), safe("ema_21"), safe("ema_50")
            features.append(1.0 if e9 > e21 else -1.0)
            features.append(1.0 if e21 > e50 else -1.0)

        if cfg.use_rsi_1m:
            features.append(self._norm_rsi(safe("rsi_1m", 50)))
        if cfg.use_rsi_5m:
            features.append(self._norm_rsi(safe("rsi_5m", 50)))
        if cfg.use_rsi_15m:
            features.append(self._norm_rsi(safe("rsi_15m", 50)))

        if cfg.use_macd_hist:
            features.append(self._norm_clamp(safe("macd_hist"), scale=0.001))
        if cfg.use_macd_cross:
            hist = safe("macd_hist")
            features.append(1.0 if hist > 0 else -1.0)

        # ── Struktura rynku ──────────────────────────────────────────
        if cfg.use_hh_hl:
            features.append(1.0 if safe("higher_highs") else -1.0)
            features.append(1.0 if safe("higher_lows", False) else -1.0)
        if cfg.use_ll_lh:
            features.append(-1.0 if safe("lower_lows") else 1.0)
            features.append(-1.0 if safe("lower_highs", False) else 1.0)
        if cfg.use_swing_dist:
            last_high = safe("last_high", price)
            last_low  = safe("last_low",  price)
            dist_high = (last_high - price) / price if price > 0 else 0
            dist_low  = (price - last_low)  / price if price > 0 else 0
            features.append(self._norm_clamp(dist_high, scale=0.05))
            features.append(self._norm_clamp(dist_low,  scale=0.05))

        # ── Wolumen ──────────────────────────────────────────────────
        if cfg.use_volume_ratio:
            features.append(self._norm_clamp(safe("volume_ratio", 1.0) - 1.0, scale=2.0))
        if cfg.use_volume_trend:
            vt = safe("volume_trend", "neutral")
            features.append(1.0 if vt == "increasing" else (-1.0 if vt == "decreasing" else 0.0))
        if cfg.use_volume_delta:
            features.append(self._norm_clamp(safe("volume_delta", 0.0), scale=1.0))
        if cfg.use_buy_volume:
            bv = safe("buy_volume", 0.5)
            features.append(self._norm_clamp(bv - 0.5, scale=0.5))
        if cfg.use_sell_volume:
            sv = safe("sell_volume", 0.5)
            features.append(self._norm_clamp(sv - 0.5, scale=0.5))
        if cfg.use_cvd:
            features.append(self._norm_clamp(safe("cvd", 0.0), scale=1.0))
        if cfg.use_large_candles:
            avg_size   = safe("avg_candle_size", 1.0)
            candle_size = abs(safe("close") - safe("open"))
            ratio = candle_size / avg_size if avg_size > 0 else 1.0
            features.append(self._norm_clamp(ratio - 1.0, scale=2.0))

        # ── Sygnały Algo #1 i #2 ─────────────────────────────────────
        if cfg.use_sentiment_score:
            features.append(safe("sentiment_score", 0.0))
        if cfg.use_trend_score_5m:
            features.append(safe("trend_score_5m", 0.0))
        if cfg.use_trend_score_15m:
            features.append(safe("trend_score_15m", 0.0))
        if cfg.use_tf_alignment:
            features.append(safe("tf_alignment", 0.0) * 2 - 1.0)

        # ── Dywergencje ──────────────────────────────────────────────
        if cfg.use_divergence_score:
            features.append(safe("divergence_score", 0.0))
        if cfg.use_divergence_bullish:
            features.append(1.0 if safe("divergence_bullish") else 0.0)
        if cfg.use_divergence_bearish:
            features.append(1.0 if safe("divergence_bearish") else 0.0)
        if cfg.use_divergence_strength:
            s = safe("divergence_strength", "none")
            features.append({"none": 0.0, "weak": 0.25, "medium": 0.65, "strong": 1.0}.get(s, 0.0))

        # ── Stan pozycji ─────────────────────────────────────────────
        if cfg.use_position_side:
            ps = safe("position_side")
            features.append(1.0 if ps == "long" else (-1.0 if ps == "short" else 0.0))
        if cfg.use_position_pnl:
            features.append(self._norm_clamp(safe("position_pnl", 0.0), scale=0.05))
        if cfg.use_position_duration:
            dur = safe("candles_in_position", 0)
            features.append(min(dur / 50.0, 1.0))
        if cfg.use_candles_since_close:
            csc = safe("candles_since_close", 0)
            features.append(min(csc / 20.0, 1.0))

        return features

    # ── Normalizacje ─────────────────────────────────────────────────

    def _norm_price_ratio(self, indicator: float, price: float) -> float:
        """Normalizuje wskaźnik cenowy jako odchylenie od ceny."""
        if price == 0:
            return 0.0
        ratio = (indicator - price) / price
        return self._norm_clamp(ratio, scale=0.02)

    def _norm_rsi(self, rsi: float) -> float:
        """RSI 0-100 → -1..+1"""
        return (rsi - 50) / 50

    def _norm_clamp(self, value: float, scale: float = 1.0) -> float:
        """Normalizuje wartość i clampuje do [-1, +1]."""
        if scale == 0:
            return 0.0
        return max(-1.0, min(1.0, value / scale))