"""Feature engineering: technical indicators computed only from past/current data.

Every feature at row t uses information available at the close of day t.
The target at row t is the log return from close t to close t+1.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100) using Wilder's exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator feature columns and the next-day ``target`` column to an OHLCV DataFrame."""
    out = df.copy()
    c = out["Close"]

    out["log_ret"] = np.log(c / c.shift(1))
    out["log_ret_5"] = np.log(c / c.shift(5))
    out["log_ret_20"] = np.log(c / c.shift(20))

    out["sma_10_ratio"] = c / c.rolling(10).mean() - 1
    out["sma_50_ratio"] = c / c.rolling(50).mean() - 1
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["ema_12_ratio"] = c / ema12 - 1
    out["ema_26_ratio"] = c / ema26 - 1

    out["rsi_14"] = _rsi(c) / 100.0
    macd = (ema12 - ema26) / c
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    out["volatility_20"] = out["log_ret"].rolling(20).std()
    mid= c.rolling(20).mean()
    std = c.rolling(20).std()
    out["bb_position"] = (c - mid) / (2 * std)

    out["hl_range"] = (out["High"] - out["Low"]) / c
    out["oc_change"] = (c - out["Open"]) / out["Open"]
    vol = out["Volume"].replace(0, np.nan)
    out["volume_change"] = np.log(vol / vol.shift(1)).fillna(0.0)
    out["day_of_week"] = out.index.dayofweek / 4.0

    # Target: next-day log return (shift -1 looks one day ahead -> label only)
    out["target"] = out["log_ret"].shift(-1)
    return out.replace([np.inf, -np.inf], np.nan)
