# data_fetcher/binance_fetcher.py
import ccxt
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}   # zmień na 'future' jeśli chcesz futures
        })

    def fetch_ohlcv_with_delta(self,
                               symbol: str = "BTC/USDT",
                               timeframe: str = "15m",
                               since: int = None,
                               limit: int = 1000,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Pobiera OHLCV + BuyVol + SellVol + VolumeDelta + CVD
        """
        logger.info(f"Pobieranie danych dla {symbol} | {timeframe} | limit={limit}")

        # Pobierz OHLCV
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Pobierz trades (dla Buy/Sell)
        df = self._add_buy_sell_volume(df, symbol)

        return df

    def _add_buy_sell_volume(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Dodaje Buy Volume, Sell Volume, Delta i CVD"""
        df = df.copy()
        df['buy_volume'] = 0.0
        df['sell_volume'] = 0.0

        for i, row in df.iterrows():
            start_time = int(row.name.timestamp() * 1000)
            end_time = int((row.name + pd.Timedelta(minutes=1)).timestamp() * 1000) if i == df.index[-1] else None

            try:
                trades = self.exchange.fetch_trades(symbol, since=start_time, limit=1000)
                time.sleep(0.1)  # rate limit

                buy_vol = 0.0
                sell_vol = 0.0

                for trade in trades:
                    trade_time = trade['timestamp']
                    if trade_time < start_time:
                        continue
                    if end_time and trade_time >= end_time:
                        break

                    amount = trade['amount']
                    if trade.get('is_buyer_maker') is False:   # agresywny kupujący
                        buy_vol += amount
                    else:                                      # agresywny sprzedający
                        sell_vol += amount

                df.loc[row.name, 'buy_volume'] = buy_vol
                df.loc[row.name, 'sell_volume'] = sell_vol

            except Exception as e:
                logger.warning(f"Nie udało się pobrać trades dla {row.name}: {e}")

        df['volume_delta'] = df['buy_volume'] - df['sell_volume']
        df['cvd'] = df['volume_delta'].cumsum()

        # Dodatkowe przydatne kolumny
        df['buy_volume_ratio'] = df['buy_volume'] / (df['buy_volume'] + df['sell_volume'] + 1e-8)

        return df

    def get_historical_data(self,
                            symbol: str,
                            timeframe: str,
                            days: int = 30) -> pd.DataFrame:
        """Wygodna metoda do pobrania większego zakresu"""
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        return self.fetch_ohlcv_with_delta(symbol, timeframe, since=since, limit=10000)