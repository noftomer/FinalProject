"""Features: daily return, volume change, and technical indicators. Every value at row t uses data up to the close of day t.

The target at row t is the log return from close t to close t+1. The LSTM sees a window of
the last ``seq_len`` rows of these features (built in dataset.py).
"""
import numpy as np
import pandas as pd

FEATURES = ["log_ret", "vol_chg", "rsi", "macd_hist", "volatility_10", "hl_range", "sma_gap_20"]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), scaled to 0..1."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return gain / (gain + loss)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the feature columns in ``FEATURES`` and the next-day ``target`` column to an OHLCV DataFrame."""
    out = df.copy()
    close = out["Close"]
    out["log_ret"] = np.log(close / close.shift(1))
    out["vol_chg"] = np.log1p(out["Volume"]).diff()                  # change in trading volume
    out["rsi"] = _rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close   # price-normalized
    out["volatility_10"] = out["log_ret"].rolling(10).std()
    out["hl_range"] = (out["High"] - out["Low"]) / close             # intraday range
    out["sma_gap_20"] = close / close.rolling(20).mean() - 1         # distance from 20-day average
    # Target: next-day log return (shift -1 looks one day ahead -> label only)
    out["target"] = out["log_ret"].shift(-1)
    return out
