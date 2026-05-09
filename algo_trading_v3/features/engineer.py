# features/engineer.py
import pandas as pd
import pandas_ta as ta
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje wszystkie techniczne cechy + bezpieczna obsługa Bollinger Bands"""
    df = df.copy()

    # Podstawowe zwroty
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Wskaźniki techniczne
    df['rsi'] = ta.rsi(df['close'], length=14)

    # MACD
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    # Bollinger Bands - BEZPIECZNA WERSJA
    bb = ta.bbands(df['close'], length=20, std=2)
    if not bb.empty and len(bb.columns) >= 3:
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    else:
        df['bb_upper'] = df['close']
        df['bb_middle'] = df['close']
        df['bb_lower'] = df['close']
        df['bb_width'] = 0.0

    # Pozostałe wskaźniki
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)

    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']

    # Czasowe
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek

    # Wypełnienie NaN - POPRAWIONE
    df = df.ffill().fillna(0)

    return df
