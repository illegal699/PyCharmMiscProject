"""
Watchdog — pilnuje volume_monitor.py
=====================================
Co minutę sprawdza czy volume_monitor działa.
Jeśli nie — wysyła alert na Telegram i restartuje skrypt.
"""

import time
import subprocess
import requests
import logging
from datetime import datetime

# ── Konfiguracja ────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = "8768725679:AAGwYOzlPwNCVLK9uxJr5HQYIIYTymcadYI"   # taki sam jak w volume_monitor.py
TELEGRAM_CHAT_ID = "5983740954"      # taki sam jak w volume_monitor.py

SCRIPT_NAME      = "volume_monitor.py"         # nazwa pilnowanego skryptu
CHECK_INTERVAL   = 60                          # sprawdzaj co ile sekund
AUTO_RESTART     = True                        # czy automatycznie restartować

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("watchdog.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

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

# ── Sprawdzanie procesu ───────────────────────────────────────────────────────

def is_running(script_name: str) -> bool:
    """Sprawdza czy skrypt jest aktywnym procesem."""
    try:
        result = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True, text=True
        )
        # Szukamy python.exe i sprawdzamy argumenty
        result2 = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "commandline", "/format:list"],
            capture_output=True, text=True
        )
        return script_name in result2.stdout
    except Exception as e:
        log.error(f"Błąd sprawdzania procesu: {e}")
        return False

def restart_script(script_name: str) -> bool:
    """Restartuje skrypt w tle."""
    try:
        subprocess.Popen(
            ["python", script_name],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        log.info(f"Zrestartowano {script_name}")
        return True
    except Exception as e:
        log.error(f"Błąd restartu: {e}")
        return False

# ── Główna pętla ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Watchdog uruchomiony")
    log.info(f"Pilnuję: {SCRIPT_NAME}")
    log.info(f"Sprawdzam co: {CHECK_INTERVAL}s")
    log.info(f"Auto-restart: {'TAK' if AUTO_RESTART else 'NIE'}")
    log.info("=" * 60)

    send_telegram(
        f"🐕 <b>Watchdog uruchomiony</b>\n"
        f"Pilnuję: <code>{SCRIPT_NAME}</code>\n"
        f"Sprawdzam co {CHECK_INTERVAL} sekund\n"
        f"Auto-restart: {'✅ TAK' if AUTO_RESTART else '❌ NIE'}"
    )

    was_running = True

    while True:
        time.sleep(CHECK_INTERVAL)
        running = is_running(SCRIPT_NAME)

        if running:
            if not was_running:
                # Skrypt wrócił do życia
                log.info(f"{SCRIPT_NAME} działa poprawnie.")
                send_telegram(f"✅ <b>Volume Monitor działa ponownie</b>")
            was_running = True

        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.warning(f"{SCRIPT_NAME} NIE działa! ({now})")

            if AUTO_RESTART:
                log.info(f"Restartuję {SCRIPT_NAME}...")
                success = restart_script(SCRIPT_NAME)
                if success:
                    send_telegram(
                        f"⚠️ <b>Volume Monitor przestał działać!</b>\n"
                        f"🔄 Automatycznie zrestartowany\n"
                        f"🕐 {now}"
                    )
                else:
                    send_telegram(
                        f"🔴 <b>Volume Monitor przestał działać!</b>\n"
                        f"❌ Restart nie powiódł się — sprawdź komputer\n"
                        f"🕐 {now}"
                    )
            else:
                send_telegram(
                    f"🔴 <b>Volume Monitor przestał działać!</b>\n"
                    f"⚠️ Uruchom go ręcznie w PyCharm\n"
                    f"🕐 {now}"
                )

            was_running = False

if __name__ == "__main__":
    main()