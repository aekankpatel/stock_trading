"""
Download and preprocess SPY data with technical indicators.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def load_data(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download OHLCV data and engineer features.
    Returns (train_df, test_df) with NaNs dropped and features normalised.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    close = df["close"]

    # Price features
    df["daily_return"] = close.pct_change()
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["ma_ratio_5_20"] = df["ma_5"] / df["ma_20"]  # momentum signal

    # Volatility
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    # RSI
    df["rsi"] = compute_rsi(close, period=14)

    # MACD
    df["macd"], df["macd_signal"] = compute_macd(close)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    df.dropna(inplace=True)
    df.reset_index(inplace=True)  # move Date into a column

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    print(f"Data loaded: {len(df)} trading days  |  train={len(train_df)}  test={len(test_df)}")
    return train_df, test_df


# Feature columns the environment will use (no price/volume raw values — agent sees normalised signals)
FEATURE_COLS = [
    "daily_return",
    "ma_ratio_5_20",
    "volatility_20",
    "rsi",
    "macd_hist",
    "volume_change",
]
