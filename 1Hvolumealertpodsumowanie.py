"""
Volume Change Monitor — Binance Perpetual + Telegram
=====================================================
Monitoruje wzrost wolumenu na parach perpetual (futures).
Automatycznie pobiera wszystkie aktywne pary USDT z Binance.
Po każdym cyklu wysyła podsumowanie — tabelę par posortowanych
od największego wzrostu wolumenu do najmniejszego.
"""

import time
import requests
import logging
from datetime import datetime

# ── Konfiguracja ────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = "8768725679:AAGwYOzlPwNCVLK9uxJr5HQYIIYTymcadYI"   # Token od @BotFather
TELEGRAM_CHAT_ID = "5983740954"      # Twoje chat_id

# Parametry
INTERVAL      = "1h"   # Timeframe świecy: 1m, 3m, 5m, 15m, 1h, 4h, 1d
UP_THRESHOLD  = 500.0    # Próg wzrostu wolumenu w % względem poprzedniej świecy

# Filtr par — zostaw puste [] żeby monitorować WSZYSTKIE pary perpetual USDT
# Lub wpisz konkretne pary żeby ograniczyć listę, np: ["BTCUSDT", "ETHUSDT"]
PAIRS_FILTER  = []

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

BINANCE_FUTURES_URL   = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

def get_all_usdt_perpetual_pairs() -> list[str]:
    try:
        r = requests.get(BINANCE_EXCHANGE_INFO, timeout=15)
        r.raise_for_status()
        data = r.json()
        pairs = [
            s["symbol"] for s in data["symbols"]
            if s["quoteAsset"] == "USDT"
            and s["contractType"] == "PERPETUAL"
            and s["status"] == "TRADING"
        ]
        pairs.sort()
        log.info(f"Pobrano {len(pairs)} aktywnych par perpetual USDT z Binance")
        return pairs
    except Exception as e:
        log.error(f"Błąd pobierania listy par: {e}")
        return []

def get_candles(symbol: str, interval: str) -> tuple[dict, dict] | None:
    try:
        r = requests.get(BINANCE_FUTURES_URL, params={
            "symbol":   symbol,
            "interval": interval,
            "limit":    3,
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()

        def parse(c):
            return {
                "time":   datetime.fromtimestamp(c[0] / 1000),
                "close":  float(c[4]),
                "volume": float(c[5]),
            }

        return parse(raw[-3]), parse(raw[-2])
    except Exception as e:
        log.error(f"Błąd pobierania danych {symbol}: {e}")
        return None

# ── Logika wskaźnika ─────────────────────────────────────────────────────────

def check_volume(symbol: str) -> dict | None:
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
    }

# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
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

def format_summary(alerts: list[dict], candle_time: datetime, total_pairs: int) -> str:
    """
    Buduje tabelę podsumowania posortowaną od największego wzrostu.
    Telegram nie obsługuje tabel HTML — używamy monospace <pre> żeby
    kolumny były równo wyrównane.
    """
    # Sortuj od największego wzrostu
    sorted_alerts = sorted(alerts, key=lambda x: x["pct_change"], reverse=True)

    # Nagłówek
    lines = [
        f"📊 <b>PODSUMOWANIE — {candle_time.strftime('%Y-%m-%d %H:%M')} ({INTERVAL})</b>",
        f"Próg: &gt;{UP_THRESHOLD:.0f}% | Spełniło warunek: <b>{len(alerts)}</b> / {total_pairs} par",
        "",
        "<pre>",
        f"{'#':<4} {'Para':<14} {'Zmiana vol':>10}",
        f"{'─'*4} {'─'*14} {'─'*10}",
    ]

    for i, a in enumerate(sorted_alerts, 1):
        symbol = a["symbol"].replace("USDT", "/USDT")
        pct_str = "+" + f"{a['pct_change']:.1f}%"
        lines.append(f"{i:<4} {symbol:<14} {pct_str:>10}")

    lines.append("</pre>")

    return "\n".join(lines)

# ── Scheduler ────────────────────────────────────────────────────────────────

INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}

def seconds_to_next_candle(interval: str) -> int:
    period_sec = INTERVAL_SECONDS.get(interval, 900)
    remaining  = period_sec - (time.time() % period_sec)
    return int(remaining) + 2

# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Volume Monitor — Binance Perpetual")
    log.info("Pobieram listę par z Binance...")

    if PAIRS_FILTER:
        pairs = PAIRS_FILTER
        log.info(f"Tryb: ręczna lista ({len(pairs)} par)")
    else:
        pairs = get_all_usdt_perpetual_pairs()
        if not pairs:
            log.error("Nie udało się pobrać listy par! Sprawdź połączenie.")
            return

    log.info(f"Monitoruję {len(pairs)} par")
    log.info(f"Interval: {INTERVAL}  |  Próg wzrostu: +{UP_THRESHOLD}%")
    log.info("=" * 60)

    send_telegram(
        f"✅ <b>Volume Monitor uruchomiony</b>\n"
        f"Monitoruję <b>{len(pairs)} par</b> perpetual USDT na Binance\n"
        f"Interwał: {INTERVAL}  |  Próg wzrostu: +{UP_THRESHOLD}%\n"
        f"Porównanie: bieżąca świeca vs poprzednia"
    )

    cycle = 0
    while True:
        wait = seconds_to_next_candle(INTERVAL)
        log.info(f"Czekam {wait}s na zamknięcie świecy ({INTERVAL})...")
        time.sleep(wait)

        cycle += 1
        if not PAIRS_FILTER and cycle % 96 == 0:
            fresh = get_all_usdt_perpetual_pairs()
            if fresh:
                pairs = fresh

        log.info(f"Sprawdzam {len(pairs)} par...")
        alerts      = []
        candle_time = None

        for symbol in pairs:
            result = check_volume(symbol)
            if result:
                alerts.append(result)
                candle_time = result["candle_time"]
                log.info(f"ALERT: {symbol} +{result['pct_change']:.1f}%")
                send_telegram(
                    f"🟢📈 <b>WZROST WOLUMENU — {result['symbol']} PERP</b>\n"
                    f"\n"
                    f"📊 Zmiana wolumenu:   <b>+{result['pct_change']:.1f}%</b>\n"
                    f"📦 Bieżąca świeca:    {result['current_vol']:,.2f}\n"
                    f"📐 Poprzednia świeca: {result['prev_vol']:,.2f}\n"
                    f"\n"
                    f"⏰ Świeca: {result['candle_time'].strftime('%Y-%m-%d %H:%M')} ({INTERVAL})\n"
                    f"🔔 Próg wzrostu: > {UP_THRESHOLD:.0f}%"
                )
            else:
                log.info(f"{symbol}: brak sygnału")
            time.sleep(0.15)

        # Wyślij podsumowanie (nawet jeśli zero par spełniło warunek)
        if candle_time is None:
            candle_time = datetime.now()

        if alerts:
            summary = format_summary(alerts, candle_time, len(pairs))
            send_telegram(summary)
            log.info(f"Podsumowanie wysłane: {len(alerts)} par spełniło warunek.")
        else:
            send_telegram(
                f"📊 <b>PODSUMOWANIE — {candle_time.strftime('%Y-%m-%d %H:%M')} ({INTERVAL})</b>\n"
                f"Żadna para nie spełniła progu &gt;{UP_THRESHOLD:.0f}%"
            )
            log.info("Podsumowanie wysłane: brak sygnałów.")

if __name__ == "__main__":
    main()