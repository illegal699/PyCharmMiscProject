"""
Volume Change Monitor — Binance Perpetual + Telegram
=====================================================
Monitoruje wzrost wolumenu na parach perpetual (futures).
Porównuje bieżącą świecę z poprzednią i wysyła alert na Telegram
gdy wolumen wzrośnie o więcej niż ustawiony próg.
"""

import time
import requests
import logging
from datetime import datetime

# ── Konfiguracja ────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = "8768725679:AAGwYOzlPwNCVLK9uxJr5HQYIIYTymcadYI"   # Token od @BotFather
TELEGRAM_CHAT_ID = "5983740954"      # Twoje chat_id

# Pary perpetual do monitorowania (symbole Binance Futures)
PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    # Dodaj kolejne pary...
]

# Parametry
INTERVAL      = "5m"   # Timeframe świecy: 1m, 3m, 5m, 15m, 1h, 4h, 1d
UP_THRESHOLD  = 50.0    # Próg wzrostu wolumenu w % względem poprzedniej świecy

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("volume_monitor.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Binance Futures API ──────────────────────────────────────────────────────

# Futures endpoint (perpetual) — inny niż spot
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

def get_candles(symbol: str, interval: str) -> tuple[dict, dict] | None:
    """
    Pobiera dwie ostatnie zamknięte świece z Binance Futures.
    Zwraca (poprzednia_świeca, bieżąca_świeca) lub None przy błędzie.
    """
    try:
        r = requests.get(BINANCE_FUTURES_URL, params={
            "symbol":   symbol,
            "interval": interval,
            "limit":    3,   # 2 zamknięte + 1 niezamknięta
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()

        # raw[-1] to niezamknięta świeca — pomijamy
        def parse(c):
            return {
                "time":   datetime.fromtimestamp(c[0] / 1000),
                "close":  float(c[4]),
                "volume": float(c[5]),
            }

        prev    = parse(raw[-3])   # świeca -2 (poprzednia)
        current = parse(raw[-2])   # świeca -1 (ostatnia zamknięta)
        return prev, current

    except Exception as e:
        log.error(f"Błąd pobierania danych {symbol}: {e}")
        return None

# ── Logika wskaźnika ─────────────────────────────────────────────────────────

def check_volume(symbol: str) -> dict | None:
    """
    Porównuje wolumen ostatniej zamkniętej świecy z poprzednią.
    Zwraca dict z wynikiem jeśli wzrost >= UP_THRESHOLD, inaczej None.
    """
    result = get_candles(symbol, INTERVAL)
    if result is None:
        return None

    prev, current = result

    if prev["volume"] == 0:
        return None

    pct_change = ((current["volume"] - prev["volume"]) / prev["volume"]) * 100

    if pct_change < UP_THRESHOLD:
        return None

    return {
        "symbol":      symbol,
        "pct_change":  pct_change,
        "current_vol": current["volume"],
        "prev_vol":    prev["volume"],
        "candle_time": current["time"],
        "close_price": current["close"],
    }

# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Wysyła wiadomość na Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Błąd Telegram: {e}")
        return False

def format_alert(result: dict) -> str:
    """Formatuje wiadomość alertu."""
    return (
        f"🟢📈 <b>WZROST WOLUMENU — {result['symbol']} PERP</b>\n"
        f"\n"
        f"💹 Cena:             <b>{result['close_price']:,.4f} USDT</b>\n"
        f"📊 Zmiana wolumenu:  <b>+{result['pct_change']:.1f}%</b>\n"
        f"📦 Bieżąca świeca:   {result['current_vol']:,.2f}\n"
        f"📐 Poprzednia świeca: {result['prev_vol']:,.2f}\n"
        f"\n"
        f"⏰ Świeca: {result['candle_time'].strftime('%Y-%m-%d %H:%M')} ({INTERVAL})\n"
        f"🔔 Próg wzrostu: > {UP_THRESHOLD:.0f}%"
    )

# ── Scheduler — czekanie na zamknięcie świecy ────────────────────────────────

INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

def seconds_to_next_candle(interval: str) -> int:
    """Oblicza ile sekund zostało do zamknięcia bieżącej świecy."""
    period_sec = INTERVAL_SECONDS.get(interval, 900)
    now        = time.time()
    remaining  = period_sec - (now % period_sec)
    return int(remaining) + 2   # +2s bufor na opóźnienie API

# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Volume Monitor — Binance Perpetual")
    log.info(f"Pary:     {', '.join(PAIRS)}")
    log.info(f"Interval: {INTERVAL}  |  Próg wzrostu: +{UP_THRESHOLD}%")
    log.info("Porównanie: bieżąca świeca vs poprzednia")
    log.info("=" * 60)

    send_telegram(
        f"✅ <b>Volume Monitor uruchomiony</b>\n"
        f"Monitoruję {len(PAIRS)} par perpetual na Binance\n"
        f"Interwał: {INTERVAL}  |  Próg wzrostu: +{UP_THRESHOLD}%\n"
        f"Porównanie: bieżąca świeca vs poprzednia"
    )

    while True:
        wait = seconds_to_next_candle(INTERVAL)
        log.info(f"Czekam {wait}s na zamknięcie świecy ({INTERVAL})...")
        time.sleep(wait)

        log.info(f"Sprawdzam {len(PAIRS)} par...")
        alerts_sent = 0

        for symbol in PAIRS:
            result = check_volume(symbol)
            if result:
                msg = format_alert(result)
                log.info(f"ALERT: {symbol} +{result['pct_change']:.1f}%")
                if send_telegram(msg):
                    alerts_sent += 1
            else:
                log.info(f"{symbol}: brak sygnału")
            time.sleep(0.2)   # małe opóźnienie żeby nie przeciążać API

        log.info(f"Wysłano {alerts_sent} alertów.")

if __name__ == "__main__":
    main()