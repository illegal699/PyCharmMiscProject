#!/usr/bin/env python3
"""
Ekonomiczny Kalendarz Bot — Telegram
- Codziennie rano (07:00 Warsaw) wysyła listę publikacji na dziś
- 30 minut przed każdą publikacją wysyła przypomnienie
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import httpx
from telegram import Bot
from telegram.constants import ParseMode

# ─────────────────────────────────────────────
# KONFIGURACJA — uzupełnij te wartości
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = "8768725679:AAGwYOzlPwNCVLK9uxJr5HQYIIYTymcadYI"        # token od @BotFather
TELEGRAM_CHAT_ID = "5983740954"         # np. "-1001234567890" lub "123456789"
FINNHUB_API_KEY = "d7sdcj9r01qorsvia8s0d7sdcj9r01qorsvia8sg"    # https://finnhub.io/register (darmowy)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")

# Minimalna ważność publikacji (1=low, 2=medium, 3=high)
MIN_IMPACT = 3

# Godzina porannego raportu (czas warszawski)
MORNING_REPORT_HOUR = 7
MORNING_REPORT_MINUTE = 0
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


async def fetch_events(target_date: date) -> list[dict]:
    """Pobiera kalendarz ekonomiczny z Finnhub dla podanej daty (tylko USA, impact >= MIN_IMPACT)."""
    date_str = target_date.strftime("%Y-%m-%d")
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": date_str,
        "to": date_str,
        "token": FINNHUB_API_KEY,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    events = data.get("economicCalendar", [])
    for e in events[:5]:
        print(e)

    IMPACT_MAP = {"low": 1, "medium": 2, "high": 3}
    filtered = []
    for ev in events:
        if ev.get("country", "").upper() != "US":
            continue
        impact_val = IMPACT_MAP.get(str(ev.get("impact", "")).lower(), 0)
        if impact_val < MIN_IMPACT:
            continue
        # Parsuj godzinę publikacji (UTC) i przelicz na Warsaw
        time_str = ev.get("time", "")  # format: "HH:MM" lub pełny datetime
        if not time_str:
            continue
        try:
            # Finnhub zwraca "YYYY-MM-DD HH:MM:SS" lub "HH:MM"
            if len(time_str) > 5:
                dt_utc = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=ZoneInfo("UTC")
                )
            else:
                dt_utc = datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=ZoneInfo("UTC"))
            dt_warsaw = dt_utc.astimezone(WARSAW_TZ)
        except ValueError:
            log.warning(f"Nie udało się sparsować czasu: {time_str}")
            continue

        filtered.append(
            {
                "event": ev.get("event", "Brak nazwy"),
                "dt_warsaw": dt_warsaw,
                "estimate": ev.get("estimate"),
                "prev": ev.get("prev"),
                "unit": ev.get("unit", ""),
            }
        )

    filtered.sort(key=lambda x: x["dt_warsaw"])
    return filtered


def format_morning_message(events: list[dict], target_date: date) -> str:
    """Formatuje poranny raport z listą publikacji."""
    date_fmt = target_date.strftime("%d.%m.%Y")
    if not events:
        return f"📅 *Kalendarz ekonomiczny — {date_fmt}*\n\nBrak ważnych publikacji US na dziś."

    lines = [f"📅 *Kalendarz ekonomiczny USA — {date_fmt}*\n"]
    for ev in events:
        hour = ev["dt_warsaw"].strftime("%H:%M")
        name = ev["event"]
        est = f"  _Konsensus: {ev['estimate']} {ev['unit']}_" if ev["estimate"] else ""
        prev = f"  _Poprzedni: {ev['prev']} {ev['unit']}_" if ev["prev"] else ""
        lines.append(f"🕐 *{hour}* — {name}{est}{prev}")

    lines.append("\n_Czas warszawski (CET/CEST)_")
    return "\n".join(lines)


def format_reminder_message(ev: dict) -> str:
    """Formatuje przypomnienie 30 minut przed publikacją."""
    hour = ev["dt_warsaw"].strftime("%H:%M")
    name = ev["event"]
    est = f"\n📊 Konsensus: *{ev['estimate']} {ev['unit']}*" if ev["estimate"] else ""
    prev = f"\n📉 Poprzedni: *{ev['prev']} {ev['unit']}*" if ev["prev"] else ""
    return (
        f"⏰ *Za 30 minut* — publikacja US:\n\n"
        f"📌 *{name}*\n"
        f"🕐 Godz. {hour}{est}{prev}"
    )


async def send_message(bot: Bot, text: str) -> None:
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
    )
    log.info(f"Wysłano wiadomość: {text[:60]}...")


async def schedule_reminders(bot: Bot, events: list[dict]) -> list[asyncio.Task]:
    """Tworzy taski z przypomnieniami 30 min przed każdą publikacją."""
    tasks = []
    now = datetime.now(tz=WARSAW_TZ)

    for ev in events:
        reminder_time = ev["dt_warsaw"] - timedelta(minutes=30)
        delay = (reminder_time - now).total_seconds()

        if delay < 0:
            log.info(f"Pominięto (za późno): {ev['event']} @ {ev['dt_warsaw'].strftime('%H:%M')}")
            continue

        async def send_reminder(e=ev, d=delay):
            await asyncio.sleep(d)
            await send_message(bot, format_reminder_message(e))

        task = asyncio.create_task(send_reminder())
        tasks.append(task)
        log.info(
            f"Zaplanowano przypomnienie: {ev['event']} "
            f"o {reminder_time.strftime('%H:%M')} (za {delay/60:.1f} min)"
        )

    return tasks


async def seconds_until(hour: int, minute: int) -> float:
    """Oblicza ile sekund zostało do kolejnego wystąpienia podanej godziny (Warsaw)."""
    now = datetime.now(tz=WARSAW_TZ)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def main_loop():
    log.info("🤖 Bot ekonomiczny uruchomiony")
    bot = Bot(token=TELEGRAM_TOKEN)

    # Przy starcie — jeśli jeszcze przed porannym raportem, wyślij od razu
    now = datetime.now(tz=WARSAW_TZ)
    morning_today = now.replace(
        hour=MORNING_REPORT_HOUR, minute=MORNING_REPORT_MINUTE, second=0, microsecond=0
    )

    reminder_tasks: list[asyncio.Task] = []

    if now < morning_today:
        # Czekaj do porannego raportu
        wait = (morning_today - now).total_seconds()
        log.info(f"Czekam {wait/60:.1f} min do porannego raportu...")
        await asyncio.sleep(wait)

    while True:
        today = datetime.now(tz=WARSAW_TZ).date()
        log.info(f"📋 Pobieram kalendarz na {today}")

        try:
            events = await fetch_events(today)
        except Exception as e:
            log.error(f"Błąd pobierania kalendarza: {e}")
            events = []

        # Wyślij poranny raport
        morning_msg = format_morning_message(events, today)
        try:
            await send_message(bot, morning_msg)
        except Exception as e:
            log.error(f"Błąd wysyłania raportu: {e}")

        # Anuluj stare taski (z poprzedniego dnia)
        for t in reminder_tasks:
            t.cancel()
        reminder_tasks = []

        # Zaplanuj przypomnienia
        if events:
            reminder_tasks = await schedule_reminders(bot, events)

        # Czekaj do jutrzejszego porannego raportu
        wait = await seconds_until(MORNING_REPORT_HOUR, MORNING_REPORT_MINUTE)
        log.info(f"Następny raport za {wait/3600:.1f} h")
        await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(main_loop())