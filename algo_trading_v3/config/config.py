# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Prosta konfiguracja bez pydantic-settings (żeby uniknąć problemów)"""

    # Binance API
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "krl1AeIWhgvoMBdRj3rNXWhlf19Vwv4v4nIrT92n1poZHZaKJdv0amEQLJwZfOLM")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "hlqnOYPIQget7NS3jMlCdcm3a2hW7NSldqfZahq2vgpgt5tephKmy5kP8lTXglPe")

    # Domyślne ustawienia
    DEFAULT_SYMBOL: str = "BTC/USDT"
    DEFAULT_TIMEFRAME: str = "15m"
    DEFAULT_DAYS: int = 30

    # Ścieżki
    DATA_DIR: str = "data/raw"
    MODELS_DIR: str = "models"


# Globalna instancja
config = Config()