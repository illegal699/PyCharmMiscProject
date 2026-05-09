import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import asyncio
import logging
from sentiment_analyzer import SentimentAnalyzer
from trend_analyzer     import TrendAnalyzer

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

async def main():
    print("\n" + "="*55)
    print("  Algo Trading — Algorytmy #1 i #2")
    print("="*55 + "\n")

    sentiment = SentimentAnalyzer()
    trend     = TrendAnalyzer(symbol="BTCUSDT")

    print("Pobieranie danych...\n")

    s_signal, t_signal = await asyncio.gather(
        sentiment.get_signal_async(),
        trend.get_signal_async(),
    )

    print("── Algorytm #1: Sentyment ──────────────────────")
    if s_signal:
        print(f"  Score:      {s_signal.composite_score:+.4f}")
        print(f"  Direction:  {s_signal.direction.value}")
        print(f"  Confidence: {s_signal.confidence:.4f}")
        print(f"  Tradeable:  {s_signal.is_tradeable}")
        fg = s_signal.fear_greed_signal
        tr = s_signal.trends_signal
        if fg: print(f"  Fear&Greed: {fg.score:+.4f}  ({fg.raw_metadata.get('classification')})")
        if tr: print(f"  Trends:     {tr.score:+.4f}  (current={tr.raw_metadata.get('current')})")
    else:
        print("  Brak sygnalu")

    print()

    print("── Algorytm #2: Trend ──────────────────────────")
    if t_signal:
        print(f"  Cena:        {t_signal.current_price}")
        print(f"  Kierunek:    {t_signal.primary_direction.value}")
        print(f"  Score:       {t_signal.score:+.4f}")
        print(f"  Strength:    {t_signal.strength.value}")
        print(f"  Confidence:  {t_signal.confidence:.4f}")
        print(f"  Alignment:   {t_signal.tf_alignment_desc}")
        print(f"  Faza:        {t_signal.market_phase.value}")
        print(f"  Strona:      {t_signal.suggested_side}")
        print(f"  Invalidation:{t_signal.invalidation_level}")
        print(f"  Tradeable:   {t_signal.is_tradeable}")
        if t_signal.skip_reason:
            print(f"  Skip:        {t_signal.skip_reason}")
    else:
        print("  Brak sygnalu")

    print()

    if s_signal and t_signal:
        print("── Ocena laczna ────────────────────────────────")
        same_dir = (
            (s_signal.composite_score > 0 and t_signal.score > 0) or
            (s_signal.composite_score < 0 and t_signal.score < 0)
        )
        if t_signal.is_tradeable and s_signal.is_tradeable and same_dir:
            print(f"  Oba algorytmy zgodne -> {t_signal.suggested_side}")
        elif t_signal.is_tradeable and not s_signal.is_tradeable:
            print(f"  Trend OK, sentyment niepewny")
        elif not t_signal.is_tradeable:
            print(f"  Trend nie-tradeable: {t_signal.skip_reason}")
        else:
            print(f"  Sygnaly sprzeczne — brak wejscia")

    sentiment.stop()
    trend.stop()

asyncio.run(main())
